"""Text metrics for the flow head.

1. Embedding fidelity (as in CoVLA): cosine of the generated MiniLM embedding
   to the true one, compared with the PCA ceiling, the mean baseline, and the
   scaled score (1 = ceiling, 0 = mean).
2. Retrieval decoding to the nearest training caption, scored by
     - exact retrieval accuracy and bank coverage
     - coarse action-class accuracy (stop / slow / accelerate / left / right /
       lane change / forward) from keywords of the action sentence
     - BLEU-4 and CIDEr-D (x100) of the retrieved action and justification.
       Note that DriveGPT4 / RAG-Driver / ADAPT generate free text while this
       decoder picks a training caption. METEOR is not computed.
"""
from __future__ import annotations

import math
import re
from collections import Counter, defaultdict

import numpy as np

def cosine(a, b):
    a = a / np.maximum(np.linalg.norm(a, axis=-1, keepdims=True), 1e-8)
    b = b / np.maximum(np.linalg.norm(b, axis=-1, keepdims=True), 1e-8)
    return (a * b).sum(-1)


def text_fidelity_report(pred_emb, true_emb, ceiling_emb, mean_emb) -> dict:
    cm = cosine(pred_emb, true_emb)
    cc = cosine(ceiling_emb, true_emb)
    cb = cosine(np.repeat(mean_emb[None], len(true_emb), 0), true_emb)
    span = float(cc.mean() - cb.mean())
    return {"n": int(len(true_emb)),
            "cosine_model": round(float(cm.mean()), 4),
            "cosine_model_median": round(float(np.median(cm)), 4),
            "cosine_pca_ceiling": round(float(cc.mean()), 4),
            "cosine_mean_baseline": round(float(cb.mean()), 4),
            "scaled_score": round((float(cm.mean()) - float(cb.mean())) / span, 4)
            if abs(span) > 1e-6 else float("nan")}


# action classes
_CLASSES = (
    ("turn_left", r"\bleft\b.*\bturn|\bturn\w*\b.*\bleft\b"),
    ("turn_right", r"\bright\b.*\bturn|\bturn\w*\b.*\bright\b"),
    ("lane_change", r"\blane\b|\bmerg|\bswerv|\bveer|\bpass(es|ing)?\b|\bovertak"),
    ("slow", r"\bslow|\bdecelerat|\bbrak"),
    ("stop", r"\bstop|\bstopped\b|\bstationary\b|\bstill\b|\bpark|\bwait|\bidl|\bhalt"),
    ("accelerate", r"\baccelerat|\bspeed(s|ing)? up\b|\bpick\w* up\b|\bstart|\bbegin|\bfaster"),
    ("forward", r"\bforward\b|\bdriv|\bmov|\bcontinu|\bsteady|\bcruis|\bgo(es|ing)?\b|\bproceed|\bstraight"),
)


def action_class(action: str) -> str:
    t = action.lower()
    for lab, pat in _CLASSES:
        if re.search(pat, t):
            return lab
    return "other"


def _tok(s: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", s.lower())


def _ngrams(toks, n):
    return Counter(tuple(toks[i:i + n]) for i in range(len(toks) - n + 1))


def bleu4(cands: list[str], refs: list[str]) -> float:
    """Corpus BLEU-4 with brevity penalty, single reference (coco-caption style)."""
    match, total = [0] * 4, [0] * 4
    c_len = r_len = 0
    for c, r in zip(cands, refs):
        ct, rt = _tok(c), _tok(r)
        c_len += len(ct)
        r_len += len(rt)
        for n in range(1, 5):
            cn, rn = _ngrams(ct, n), _ngrams(rt, n)
            match[n - 1] += sum(min(v, rn[g]) for g, v in cn.items())
            total[n - 1] += max(len(ct) - n + 1, 0)
    if min(total) == 0 or min(match) == 0:
        return 0.0
    logp = sum(math.log(m / t) for m, t in zip(match, total)) / 4
    bp = 1.0 if c_len > r_len else math.exp(1 - r_len / max(c_len, 1))
    return 100 * bp * math.exp(logp)


def cider_d(cands: list[str], refs: list[str], sigma: float = 6.0) -> float:
    """CIDEr-D as in pycocoevalcap, single reference, reported x100."""
    ref_counts = [Counter() for _ in refs]
    cand_counts = [Counter() for _ in cands]
    for i, (c, r) in enumerate(zip(cands, refs)):
        ct, rt = _tok(c), _tok(r)
        for n in range(1, 5):
            ref_counts[i].update(_ngrams(rt, n))
            cand_counts[i].update(_ngrams(ct, n))
    df = defaultdict(float)
    for rc in ref_counts:
        for g in rc:
            df[g] += 1
    log_n = math.log(float(len(refs)))

    def vec(cnt):
        v = [dict() for _ in range(4)]
        norm = [0.0] * 4
        length = 0
        for g, tf in cnt.items():
            n = len(g) - 1
            v[n][g] = tf * (log_n - math.log(max(1.0, df[g])))
            norm[n] += v[n][g] ** 2
            if n == 1:
                length += tf
        return v, [math.sqrt(x) for x in norm], length

    scores = []
    for cc, rc in zip(cand_counts, ref_counts):
        vh, nh, lh = vec(cc)
        vr, nr, lr = vec(rc)
        val = 0.0
        for n in range(4):
            s = sum(min(x, vr[n].get(g, 0.0)) * vr[n].get(g, 0.0) for g, x in vh[n].items())
            if nh[n] and nr[n]:
                s /= nh[n] * nr[n]
            val += s * math.exp(-((lh - lr) ** 2) / (2 * sigma ** 2))
        scores.append(val / 4 * 10.0)
    return 100 * float(np.mean(scores))


class CaptionBank:
    """Unique training captions with their embeddings and action / justification parts."""

    def __init__(self, train):
        seen = {}
        for t, e, a, j in zip(train.text, train.emb, train.action, train.justification):
            seen.setdefault(t, (e, a, j))
        self.text = list(seen)
        self.emb = np.stack([seen[t][0] for t in self.text])
        self.action = [seen[t][1] for t in self.text]
        self.justification = [seen[t][2] for t in self.text]
        self.emb_n = self.emb / np.maximum(np.linalg.norm(self.emb, axis=1, keepdims=True), 1e-8)

    def nearest(self, pred_emb, chunk: int = 4096):
        p = pred_emb / np.maximum(np.linalg.norm(pred_emb, axis=1, keepdims=True), 1e-8)
        out = np.empty(len(p), dtype=np.int64)
        for i in range(0, len(p), chunk):
            out[i:i + chunk] = (p[i:i + chunk] @ self.emb_n.T).argmax(1)
        return out


def decode_report(pred_emb, split, bank: CaptionBank, caption_metrics: bool = True) -> dict:
    nn_idx = bank.nearest(pred_emb)
    r_text = [bank.text[i] for i in nn_idx]
    r_act = [bank.action[i] for i in nn_idx]
    r_just = [bank.justification[i] for i in nn_idx]
    bank_set = set(bank.text)
    covered = np.array([t in bank_set for t in split.text])
    exact = np.array([a == b for a, b in zip(r_text, split.text)])
    gt_cls = [action_class(a) for a in split.action]
    pr_cls = [action_class(a) for a in r_act]
    rep = {"bank_size": len(bank.text),
           "retrieval_accuracy": round(float(exact.mean()), 4),
           "bank_coverage": round(float(covered.mean()), 4),
           "retrieved_caption_cosine": round(float(cosine(bank.emb[nn_idx], split.emb).mean()), 4),
           "action_class_accuracy": round(float(np.mean([a == b for a, b in zip(gt_cls, pr_cls)])), 4)}
    if caption_metrics:
        rep["action_bleu4"] = round(bleu4(r_act, list(split.action)), 2)
        rep["action_cider"] = round(cider_d(r_act, list(split.action)), 2)
        rep["justification_bleu4"] = round(bleu4(r_just, list(split.justification)), 2)
        rep["justification_cider"] = round(cider_d(r_just, list(split.justification)), 2)
    return rep


def class_distribution(actions) -> dict:
    c = Counter(action_class(a) for a in actions)
    n = sum(c.values())
    return {k: round(v / n, 4) for k, v in c.most_common()}
