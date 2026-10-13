"""Frozen encoders, both cached to disk.

* image: ImageNet MobileNetV3-Small, average-pooled 576-d features (as in
  CoVLA/vla_trajecotry_head/encoders.py).
* text : all-MiniLM-L6-v2 (384-d, L2-normalised), the same encoder used for
  CoVLA's precomputed caption embeddings.
"""
from __future__ import annotations

import hashlib
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
        # plain resize to 224x224 (aspect ratio not kept, same as CoVLA)
        img = Image.open(self.paths[i]).convert("RGB").resize(
            (self.size, self.size), Image.BILINEAR)
        x = torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).permute(2, 0, 1)
        return x, i


@torch.no_grad()
def extract_image_features(image_paths: list[Path], cfg: Config, cache_path: Path,
                           device: str = "cpu", force: bool = False) -> np.ndarray:
    """(N, 576) pooled features, one row per path, in the given order. Cached."""
    path_strs = np.array([Path(p).name for p in image_paths])

    if cache_path.exists() and not force:
        blob = np.load(cache_path, allow_pickle=True)
        if blob["paths"].shape == path_strs.shape and np.array_equal(blob["paths"], path_strs):
            print(f"  [cache] image features {blob['feat'].shape} <- {cache_path.name}")
            return blob["feat"]
        print(f"  [{cache_path.name}] cached image selection is stale -> re-extracting")

    backbone = build_backbone(cfg).to(device)
    mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
    dl = DataLoader(_ImageFileDataset(image_paths, cfg.image_size), batch_size=64,
                    shuffle=False, num_workers=cfg.num_workers)

    feats = np.zeros((len(image_paths), cfg.img_feat_dim), dtype=np.float32)
    n_done = 0
    for x, idx in dl:
        x = (x.to(device) - mean) / std
        feats[idx.numpy()] = backbone(x).mean(dim=(2, 3)).cpu().numpy()
        n_done += len(idx)
        if n_done % (64 * 20) == 0:
            print(f"    ...{n_done}/{len(image_paths)} images encoded")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, feat=feats, paths=path_strs)
    print(f"  [cnn] cached {feats.shape} -> {cache_path}")
    return feats


def _text_key(texts: list[str], model: str) -> str:
    h = hashlib.sha1(model.encode())
    for t in texts:
        h.update(b"\x00" + t.encode("utf-8"))
    return h.hexdigest()[:16]


def embed_texts(texts: list[str], cfg: Config, cache_path: Path,
                device: str = "cpu", force: bool = False) -> np.ndarray:
    """(len(texts), 384) L2-normalised MiniLM embeddings of unique strings. Cached."""
    key = _text_key(texts, cfg.text_model)
    if cache_path.exists() and not force:
        blob = np.load(cache_path, allow_pickle=True)
        if str(blob["key"]) == key:
            print(f"  [cache] text embeddings {blob['emb'].shape} <- {cache_path.name}")
            return blob["emb"]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit("pip install sentence-transformers  (needed once, to embed the "
                         "DriveLM captions with all-MiniLM-L6-v2)") from e
    print(f"  [text] embedding {len(texts)} unique strings with {cfg.text_model}")
    model = SentenceTransformer(cfg.text_model, device=device)
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    np.savez_compressed(cache_path, emb=emb, key=key)
    return emb
