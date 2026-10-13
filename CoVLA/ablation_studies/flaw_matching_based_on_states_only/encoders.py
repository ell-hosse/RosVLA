"""Frozen image encoder: JPEG frame -> pooled MobileNetV3-Small feature (cached).

Caption embeddings are precomputed by covla_preprocessed
(cache_2hz/capemb_minilm_384.npz), so there is no text encoder here.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from config import Config

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_backbone(cfg: Config):
    import torchvision.models as tvm

    if cfg.cnn_name != "mobilenet_v3_small":
        raise ValueError(f"unsupported cnn_name {cfg.cnn_name!r}")
    net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.features


class _ImageFileDataset(Dataset):
    def __init__(self, paths: list[Path], size: int):
        self.paths = paths
        self.size = size

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR
        )
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, i


@torch.no_grad()
def extract_image_features(
    image_paths: list[Path], cfg: Config, cache_path: Path,
    device: str = "cpu", force: bool = False,
) -> np.ndarray:
    """(N, img_feat_dim) pooled features, one row per path, in the given order. Cached."""
    path_strs = np.array([str(p) for p in image_paths])

    if cache_path.exists() and not force:
        blob = np.load(cache_path, allow_pickle=True)
        if blob["paths"].shape == path_strs.shape and np.array_equal(blob["paths"], path_strs):
            print(f"  [cache] image features {blob['feat'].shape} <- {cache_path.name}")
            return blob["feat"]
        print(f"  [{cache_path.name}] cached image selection is stale -> re-extracting")

    if len(image_paths) == 0:
        feats = np.zeros((0, cfg.img_feat_dim), dtype=np.float32)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, feat=feats, paths=path_strs)
        return feats

    backbone = build_backbone(cfg).to(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)

    ds = _ImageFileDataset(image_paths, cfg.image_size)
    dl = DataLoader(ds, batch_size=128, shuffle=False, num_workers=cfg.num_workers)

    feats = np.zeros((len(image_paths), cfg.img_feat_dim), dtype=np.float32)
    n_done = 0
    for x, idx in dl:
        x = x.to(device)
        x = (x - mean) / std
        fmap = backbone(x)                 # (b, 576, 7, 7) for a 224 input
        pooled = fmap.mean(dim=(2, 3))      # global average pool -> (b, 576)
        feats[idx.numpy()] = pooled.cpu().numpy()
        n_done += len(idx)
        if n_done % (128 * 20) == 0:
            print(f"    ...{n_done}/{len(image_paths)} images encoded")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feat=feats, paths=path_strs)
    print(f"  [cnn] cached {feats.shape} -> {cache_path}")
    return feats
