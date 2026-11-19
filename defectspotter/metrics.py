"""Ranking and localisation metrics: image AUROC/AP, pixel AUROC, AUPRO, bootstrap CIs."""
from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve


def safe_auroc(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, np.asarray(scores, dtype=np.float64)))


def safe_ap(labels: Sequence[int], scores: Sequence[float]) -> float:
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, np.asarray(scores, dtype=np.float64)))


def bootstrap_auroc_ci(labels, scores, n_boot: int = 1000, confidence: float = 0.95,
                       seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling good and defective parts separately (stratified)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    pos, neg = np.flatnonzero(labels == 1), np.flatnonzero(labels == 0)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    estimates = np.empty(n_boot)
    for b in range(n_boot):
        p = scores[rng.choice(pos, len(pos))]
        n = scores[rng.choice(neg, len(neg))]
        # Mann-Whitney form of AUROC (fast, tie-aware)
        estimates[b] = (np.mean(p[:, None] > n[None, :]) + 0.5 * np.mean(p[:, None] == n[None, :]))
    alpha = (1 - confidence) / 2
    return float(np.quantile(estimates, alpha)), float(np.quantile(estimates, 1 - alpha))


def pixel_auroc(maps: Sequence[np.ndarray], masks: Sequence[np.ndarray], max_pixels: int = 4_000_000,
                seed: int = 42) -> float:
    scores = np.concatenate([np.asarray(m, np.float32).ravel() for m in maps])
    labels = np.concatenate([(np.asarray(m) > 0).ravel() for m in masks]).astype(np.uint8)
    if labels.min() == labels.max():
        return float("nan")
    if len(scores) > max_pixels:
        keep = np.random.default_rng(seed).choice(len(scores), max_pixels, replace=False)
        scores, labels = scores[keep], labels[keep]
    return float(roc_auc_score(labels, scores))


def aupro(maps: Sequence[np.ndarray], masks: Sequence[np.ndarray], fpr_limit: float = 0.3,
          num_thresholds: int = 400) -> float:
    """Area under the per-region-overlap curve up to ``fpr_limit``, normalised to [0,1].

    Every connected defect region counts equally, so small defects matter as much
    as large ones (Bergmann et al., MVTec AD). Vectorised with sorted scores.
    """
    maps = [np.asarray(m, np.float32) for m in maps]
    masks = [(np.asarray(m) > 0).astype(np.uint8) for m in masks]
    negatives = np.sort(np.concatenate([m[k == 0] for m, k in zip(maps, masks)]))
    regions: list[np.ndarray] = []
    for anomaly_map, mask in zip(maps, masks):
        if mask.any():
            count, labels = cv2.connectedComponents(mask, connectivity=8)
            for region_id in range(1, count):
                regions.append(np.sort(anomaly_map[labels == region_id]))
    if not regions or len(negatives) == 0:
        return float("nan")
    # Candidate thresholds: every score region that can produce FPR <= limit (negatives in the upper tail
    # plus all defect-region scores), sampled by quantile, plus the point just above the largest negative
    # (FPR = 0) so perfectly separating maps are handled exactly.
    cut = float(np.quantile(negatives, max(0.0, 1.0 - fpr_limit * 1.2)))
    candidates = np.concatenate([negatives[negatives >= cut], *[r[r >= cut] for r in regions]])
    thresholds = np.quantile(candidates, np.linspace(0.0, 1.0, num_thresholds)) if len(candidates) else np.array([cut])
    thresholds = np.unique(np.r_[thresholds, np.nextafter(negatives[-1], np.inf)])[::-1]
    fpr = 1.0 - np.searchsorted(negatives, thresholds, side="left") / len(negatives)
    pro = np.zeros(len(thresholds))
    for region in regions:
        pro += 1.0 - np.searchsorted(region, thresholds, side="left") / len(region)
    pro /= len(regions)
    order = np.argsort(fpr)
    fpr, pro = np.concatenate([[0.0], fpr[order]]), np.concatenate([[0.0], pro[order]])
    keep = fpr <= fpr_limit
    f, p = fpr[keep], pro[keep]
    if f[-1] < fpr_limit:
        f = np.append(f, fpr_limit)
        p = np.append(p, np.interp(fpr_limit, fpr, pro))
    trapezoid = getattr(np, "trapezoid", None) or np.trapz
    return float(trapezoid(p, f) / fpr_limit)


def roc_points(labels, scores, max_points: int = 120) -> dict[str, list[float]]:
    """Down-sampled ROC curve for interactive charts."""
    labels = np.asarray(labels).astype(int)
    if len(np.unique(labels)) < 2:
        return {"fpr": [], "tpr": []}
    fpr, tpr, _ = roc_curve(labels, np.asarray(scores, np.float64))
    if len(fpr) > max_points:
        idx = np.unique(np.linspace(0, len(fpr) - 1, max_points).astype(int))
        fpr, tpr = fpr[idx], tpr[idx]
    return {"fpr": fpr.round(5).tolist(), "tpr": tpr.round(5).tolist()}
