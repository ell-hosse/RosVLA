"""Evaluation metrics: trajectory (ADE/FDE) and text-embedding fidelity."""

from __future__ import annotations

import numpy as np


def ade_fde(pred: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """ADE and FDE for (N, T, 2) trajectories in metres."""
    d = np.linalg.norm(pred - true, axis=-1)          # (N, T)
    return float(d.mean()), float(d[:, -1].mean())


def min_ade_fde(preds: np.ndarray, true: np.ndarray) -> tuple[float, float]:
    """Best-of-K ADE/FDE. preds: (K, N, T, 2)."""
    d = np.linalg.norm(preds - true[None], axis=-1)   # (K, N, T)
    ade = d.mean(axis=2).min(axis=0)                  # (N,)
    fde = d[:, :, -1].min(axis=0)                      # (N,)
    return float(ade.mean()), float(fde.mean())


def cosine(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return (a * b).sum(-1)


def text_fidelity_report(
    pred_emb: np.ndarray,      # (N, 384) generated (flow z, decoded back through PCA)
    true_emb: np.ndarray,      # (N, 384) real caption embedding
    ceiling_emb: np.ndarray,   # (N, 384) true embedding round-tripped through the PCA
    mean_emb: np.ndarray,      # (384,) train-mean caption embedding
) -> dict:
    """Cosine between generated and true caption embeddings, with reference points.

    cosine_pca_ceiling is the PCA reconstruction (best achievable with this
    latent) and cosine_mean_baseline always predicts the mean training caption.
    scaled_score maps cosine_model so that 1 = ceiling and 0 = mean baseline.
    """
    cos_model = cosine(pred_emb, true_emb)
    cos_ceiling = cosine(ceiling_emb, true_emb)
    cos_mean = cosine(np.repeat(mean_emb[None], len(true_emb), axis=0), true_emb)

    span = float(cos_ceiling.mean() - cos_mean.mean())
    scaled = ((float(cos_model.mean()) - float(cos_mean.mean())) / span
              if abs(span) > 1e-6 else float("nan"))

    return {
        "n": int(len(true_emb)),
        "cosine_model": round(float(cos_model.mean()), 4),
        "cosine_pca_ceiling": round(float(cos_ceiling.mean()), 4),
        "cosine_mean_baseline": round(float(cos_mean.mean()), 4),
        "scaled_score": round(scaled, 4),
    }


def retrieval_report(
    pred_emb: np.ndarray, true_texts: list[str],
    bank_texts: list[str], bank_emb: np.ndarray,
) -> dict:
    """Nearest-neighbour caption retrieval accuracy against a sampled caption bank.

    Use a size-capped bank (--retrieval-bank-size); the full training vocabulary
    is too large.
    """
    bank_n = bank_emb / np.maximum(np.linalg.norm(bank_emb, axis=1, keepdims=True), 1e-8)
    pred_n = pred_emb / np.maximum(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-8)
    sim = pred_n @ bank_n.T
    nearest = sim.argmax(axis=1)
    retrieved = [bank_texts[i] for i in nearest]
    bank_set = set(bank_texts)
    covered = np.array([t in bank_set for t in true_texts])
    correct = np.array([r == t for r, t in zip(retrieved, true_texts)])
    return {
        "bank_size": len(bank_texts),
        "retrieval_accuracy": round(float(correct.mean()), 4),
        "retrieval_accuracy_covered": (
            round(float(correct[covered].mean()), 4) if covered.any() else None
        ),
        "bank_coverage": round(float(covered.mean()), 4),
    }
