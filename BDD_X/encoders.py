"""Frozen encoders, both cached to disk.

* frames: each clip mp4 (32 frames, 455x256) is decoded with OpenCV and the 8
  frames at CACHED_FRAME_IDX go through MobileNetV3-Small -> (N, 8, 576),
  cached as float16. --n-frames selects from the cache. Undecodable mp4s are
  dropped from all ablations.
* text: all-MiniLM-L6-v2, L2-normalised 384-d (as for CoVLA and DriveLM).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from config import CACHED_FRAME_IDX, Config

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def build_backbone(cfg: Config):
    import torchvision.models as tvm
    net = tvm.mobilenet_v3_small(weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net.features


def read_frames(mp4: Path, size: int) -> np.ndarray | None:
    """Return (8, size, size, 3) float32 RGB frames at CACHED_FRAME_IDX, or None if
    the file cannot be decoded.
    """
    import cv2
    try:
        cap = cv2.VideoCapture(str(mp4))
        frames = []
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            frames.append(fr)
        cap.release()
    except Exception:
        return None
    if not frames:
        return None
    n = len(frames)
    idx = [min(n - 1, int(round(i * (n - 1) / 31))) for i in CACHED_FRAME_IDX]
    out = np.stack([cv2.resize(cv2.cvtColor(frames[i], cv2.COLOR_BGR2RGB), (size, size),
                               interpolation=cv2.INTER_LINEAR) for i in idx])
    return out.astype(np.float32) / 255.0


class _Clips(Dataset):
    def __init__(self, paths, size, todo):
        self.paths, self.size, self.todo = paths, size, todo

    def __len__(self):
        return len(self.todo)

    def __getitem__(self, j):
        i = int(self.todo[j])
        fr = read_frames(self.paths[i], self.size)
        ok = fr is not None
        if not ok:
            fr = np.zeros((len(CACHED_FRAME_IDX), self.size, self.size, 3), np.float32)
        return torch.from_numpy(fr).permute(0, 3, 1, 2), i, ok


@torch.no_grad()
def extract_clip_features(paths: list[Path], names: list[str], cfg: Config, cache_path: Path,
                          device: str = "cpu", force: bool = False):
    """Return (feat (N, 8, 576) float16, ok (N,) bool), one row per clip.

    Undecodable clips get ok=False and are listed in cache/undecodable_clips.txt.
    Progress is checkpointed to <cache>.partial.npz so a run can resume.
    """
    names = np.array(names)
    if cache_path.exists() and not force:
        blob = np.load(cache_path, allow_pickle=True)
        if (blob["names"].shape == names.shape and np.array_equal(blob["names"], names)
                and "ok" in blob.files):
            print(f"  [cache] clip features {blob['feat'].shape} "
                  f"({int((~blob['ok']).sum())} undecodable) <- {cache_path.name}")
            return blob["feat"], blob["ok"]
        print(f"  [{cache_path.name}] cached clip selection is stale -> re-extracting")

    partial = cache_path.with_name(cache_path.stem + ".partial.npz")
    feats = np.zeros((len(paths), len(CACHED_FRAME_IDX), cfg.img_feat_dim), np.float16)
    ok = np.zeros(len(paths), bool)
    done = np.zeros(len(paths), bool)
    if partial.exists() and not force:
        blob = np.load(partial, allow_pickle=True)
        if blob["names"].shape == names.shape and np.array_equal(blob["names"], names):
            feats, ok, done = blob["feat"], blob["ok"], blob["done"]
            print(f"  [resume] {int(done.sum())}/{len(paths)} clips already encoded")

    todo = np.where(~done)[0]
    if len(todo):
        backbone = build_backbone(cfg).to(device)
        mean, std = IMAGENET_MEAN.to(device), IMAGENET_STD.to(device)
        dl = DataLoader(_Clips(paths, cfg.image_size, todo), batch_size=16, shuffle=False,
                        num_workers=cfg.num_workers)
        n_seen, last_ckpt = int(done.sum()), int(done.sum())
        for x, idx, good in dl:                      # x: (b, 8, 3, H, W)
            b, k = x.shape[:2]
            x = (x.view(b * k, *x.shape[2:]).to(device) - mean) / std
            f = backbone(x).mean(dim=(2, 3)).view(b, k, -1)
            idx = idx.numpy()
            feats[idx] = f.cpu().numpy().astype(np.float16)
            ok[idx] = good.numpy()
            done[idx] = True
            for i in idx[~good.numpy()]:
                print(f"    [skip] cannot decode {paths[i]}")
            n_seen += b
            if n_seen - last_ckpt >= 2000:
                np.savez(partial, feat=feats, ok=ok, done=done, names=names)
                last_ckpt = n_seen
                print(f"    ...{n_seen}/{len(paths)} clips encoded (checkpointed)")

    np.savez(cache_path, feat=feats, ok=ok, names=names)
    bad = [str(paths[i]) for i in np.where(~ok)[0]]
    (cache_path.parent / "undecodable_clips.txt").write_text("\n".join(bad))
    if partial.exists():
        try:
            partial.unlink()
        except OSError:
            pass
    print(f"  [cnn] cached {feats.shape} -> {cache_path}; {len(bad)} undecodable clip(s) "
          f"listed in undecodable_clips.txt")
    return feats, ok


def embed_texts(texts: list[str], cfg: Config, cache_path: Path, device: str = "cpu") -> np.ndarray:
    """(len(texts), 384) L2-normalised MiniLM embeddings of unique strings. Cached."""
    h = hashlib.sha1(cfg.text_model.encode())
    for t in texts:
        h.update(b"\x00" + t.encode("utf-8"))
    key = h.hexdigest()[:16]
    if cache_path.exists():
        blob = np.load(cache_path, allow_pickle=True)
        if str(blob["key"]) == key:
            print(f"  [cache] text embeddings {blob['emb'].shape} <- {cache_path.name}")
            return blob["emb"]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as e:
        raise SystemExit("pip install sentence-transformers") from e
    print(f"  [text] embedding {len(texts)} unique strings with {cfg.text_model}")
    model = SentenceTransformer(cfg.text_model, device=device)
    emb = model.encode(texts, batch_size=256, convert_to_numpy=True,
                       normalize_embeddings=True, show_progress_bar=True).astype(np.float32)
    np.savez_compressed(cache_path, emb=emb, key=key)
    return emb
