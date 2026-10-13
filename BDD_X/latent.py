"""PCA text latent, fitted on the training caption embeddings.

Same as CoVLA/vla_trajecotry_head/latent.py.
"""
from __future__ import annotations

import numpy as np


class TextLatent:
    def __init__(self, train_emb: np.ndarray, max_dim: int):
        uniq = np.unique(train_emb, axis=0)
        rank_cap = max(1, len(uniq) - 1)
        self.dim = int(min(max_dim, rank_cap, train_emb.shape[1]))
        self.n_unique_train = int(len(uniq))

        self.mean = uniq.mean(axis=0, keepdims=True).astype(np.float32)
        centred = uniq - self.mean
        _, sv, vt = np.linalg.svd(centred, full_matrices=False)
        self.basis = vt[: self.dim].astype(np.float32)

        total = float((sv ** 2).sum()) or 1.0
        self.explained = float((sv[: self.dim] ** 2).sum() / total)
        self.explained_per_component = ((sv[: self.dim] ** 2) / total).astype(np.float32)

        codes = centred @ self.basis.T
        self.scale = np.maximum(codes.std(axis=0, keepdims=True), 1e-6).astype(np.float32)

    def encode(self, emb: np.ndarray) -> np.ndarray:
        return ((emb - self.mean) @ self.basis.T) / self.scale

    def decode(self, code: np.ndarray) -> np.ndarray:
        out = (code * self.scale) @ self.basis + self.mean
        norm = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.maximum(norm, 1e-8)

    def reconstruct(self, emb: np.ndarray) -> np.ndarray:
        return self.decode(self.encode(emb))


def pca_report(latent: TextLatent, splits: dict[str, np.ndarray]) -> dict:
    report = {
        "latent_dim": latent.dim,
        "n_unique_train_texts": latent.n_unique_train,
        "explained_variance_ratio_cumulative_train": round(latent.explained, 4),
        "explained_variance_ratio_per_component": [
            round(float(x), 5) for x in latent.explained_per_component],
    }
    for name, emb in splits.items():
        if emb is None or len(emb) == 0:
            continue
        recon = latent.reconstruct(emb)
        report[name] = {
            "n": int(len(emb)),
            "n_unique": int(len(np.unique(emb, axis=0))),
            "reconstruction_mse": round(float(np.mean((recon - emb) ** 2)), 6),
            "reconstruction_cosine": round(float((recon * emb).sum(-1).mean()), 4),
        }
    return report
