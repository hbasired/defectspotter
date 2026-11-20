"""From anomaly scores to factory decisions: scaling, PASS/CHECK/REJECT and costs.

Score scale (0-100, label-free)
    25 = median score of held-out GOOD parts
    75 = median score of SYNTHETIC defects
so an operator can read any score against "typical good" and "typical defect"
without a single real defect having been used.

Pixel scale (0-100) for heat maps
     0 = median pixel of good parts, 50 = 99.5th percentile of good pixels.
Pixels above 50 are more unusual than 99.5% of good-part pixels.

Three-way rule with threshold t and half-band b:
    PASS   score <  t - b          (ship)
    CHECK  t - b <= score < t + b  (human inspector decides)
    REJECT score >= t + b          (quarantine / rework)
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

THRESHOLD_GRID = np.round(np.arange(0.0, 100.01, 0.5), 2)


@dataclass
class CostModel:
    """Illustrative economics (editable in the app). Units: currency per part."""

    defect_prevalence: float = 0.01   # 1% of produced parts are defective
    cost_escape: float = 400.0        # defective part reaches the customer (warranty, complaint, recall share)
    cost_false_reject: float = 2.0    # good part scrapped or reworked
    cost_review: float = 0.30         # ~20 s of an inspector's time for a CHECK
    review_capacity: float = 0.10     # at most 10% of parts may be routed to CHECK
    human_detection_rate: float = 0.80  # 100% manual inspection catches ~80% of defects (literature range 70-90%)
    parts_per_day: int = 20000

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------- scaling
def fit_score_scale(good_scores, synthetic_scores) -> dict:
    good = np.asarray(good_scores, np.float64)
    synthetic = np.asarray(synthetic_scores, np.float64)
    good_median = float(np.median(good))
    syn_median = float(np.median(synthetic)) if len(synthetic) else good_median
    spread = max(syn_median - good_median, 4.0 * float(np.std(good)), 1e-9)
    return {"good_median": good_median, "synthetic_median": syn_median, "spread": spread,
            "anchor_good": 25.0, "anchor_synthetic": 75.0}


def normalize_scores(raw, scale: dict) -> np.ndarray:
    raw = np.asarray(raw, np.float64)
    return 25.0 + 50.0 * (raw - scale["good_median"]) / scale["spread"]


def fit_pixel_scale(good_maps, quantile: float = 0.999, max_values: int = 2_000_000, seed: int = 0,
                    synthetic_maps=None, synthetic_masks=None) -> dict:
    """Piecewise-linear pixel scale, label-free:

    0   = median pixel of held-out good parts
    50  = ``quantile`` of good pixels (only 0.1% of normal pixels are brighter)
    100 = median pixel *inside* synthetic-defect masks (typical defect strength)
    """
    values = np.concatenate([np.asarray(m, np.float32).ravel() for m in good_maps])
    if len(values) > max_values:
        values = values[np.random.default_rng(seed).choice(len(values), max_values, replace=False)]
    low = float(np.median(values))
    threshold = max(float(np.quantile(values, quantile)), low + 1e-9)
    high = threshold + (threshold - low)
    if synthetic_maps is not None and synthetic_masks is not None:
        inside = [np.asarray(m, np.float32)[np.asarray(k) > 0] for m, k in zip(synthetic_maps, synthetic_masks)]
        inside = np.concatenate([v for v in inside if v.size]) if any(v.size for v in inside) else np.empty(0)
        if inside.size:
            high = max(high, float(np.median(inside)))
    return {"low": low, "threshold": threshold, "high": high, "quantile": quantile}


def normalize_map(anomaly_map, scale: dict) -> np.ndarray:
    values = np.asarray(anomaly_map, np.float32)
    low, threshold = scale["low"], scale["threshold"]
    high = scale.get("high", threshold + (threshold - low))
    below = 50.0 * (values - low) / (threshold - low)
    above = 50.0 + 50.0 * (values - threshold) / max(high - threshold, 1e-9)
    return np.where(values <= threshold, below, above).astype(np.float32)


# ------------------------------------------------------------------ decisions
def decide(score: float, threshold: float, band: float) -> str:
    if score < threshold - band:
        return "PASS"
    if score < threshold + band:
        return "CHECK"
    return "REJECT"


def decide_many(scores, threshold: float, band: float) -> np.ndarray:
    scores = np.asarray(scores, np.float64)
    out = np.full(scores.shape, "CHECK", dtype=object)
    out[scores < threshold - band] = "PASS"
    out[scores >= threshold + band] = "REJECT"
    return out


def operating_metrics(scores, labels, threshold: float, band: float, costs: CostModel) -> dict:
    """Rates on a labelled set plus prevalence-adjusted expected cost per 1,000 parts."""
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels).astype(int)
    good, bad = scores[labels == 0], scores[labels == 1]
    lo, hi = threshold - band, threshold + band
    frr = float(np.mean(good >= hi)) if len(good) else float("nan")            # good -> REJECT
    good_check = float(np.mean((good >= lo) & (good < hi))) if len(good) else float("nan")
    escape = float(np.mean(bad < lo)) if len(bad) else float("nan")            # defect -> PASS
    bad_check = float(np.mean((bad >= lo) & (bad < hi))) if len(bad) else float("nan")
    detect = float(np.mean(bad >= hi)) if len(bad) else float("nan")
    p = costs.defect_prevalence
    review = (1 - p) * np.nan_to_num(good_check) + p * np.nan_to_num(bad_check)
    fr_cost = 1000 * (1 - p) * np.nan_to_num(frr) * costs.cost_false_reject
    esc_cost = 1000 * p * np.nan_to_num(escape) * costs.cost_escape
    rev_cost = 1000 * review * costs.cost_review
    # Precision of an automatic REJECT at real-world prevalence (Bayes).
    tpr_reject = np.nan_to_num(detect)
    ppv = (p * tpr_reject) / max(p * tpr_reject + (1 - p) * np.nan_to_num(frr), 1e-12)
    return {
        "threshold": float(threshold), "band": float(band),
        "false_reject_rate": frr, "escape_rate": escape,
        "good_check_rate": good_check, "defect_check_rate": bad_check, "defect_reject_rate": detect,
        "review_rate": float(review),
        "reject_precision_at_prevalence": float(ppv),
        "false_reject_cost_per_1000": float(fr_cost), "escape_cost_per_1000": float(esc_cost),
        "review_cost_per_1000": float(rev_cost), "expected_cost_per_1000": float(fr_cost + esc_cost + rev_cost),
        "n_good": int(len(good)), "n_defect": int(len(bad)),
    }


def cost_curve(scores, labels, band: float, costs: CostModel, thresholds=THRESHOLD_GRID) -> pd.DataFrame:
    return pd.DataFrame([operating_metrics(scores, labels, t, band, costs) for t in thresholds])


def kde_bandwidth(samples) -> float:
    """Silverman's rule of thumb (robust spread), floored so a tight cluster still gets a smooth tail."""
    x = np.asarray(samples, np.float64)
    if len(x) < 2:
        return 1.0
    spread = min(float(np.std(x, ddof=1)), float(np.subtract(*np.percentile(x, [75, 25]))) / 1.34)
    if spread <= 0:
        spread = float(np.std(x, ddof=1)) or 1.0
    return max(0.9 * spread * len(x) ** -0.2, 0.5)


def kde_cdf(samples, points, bandwidth: float | None = None) -> np.ndarray:
    """P(score < point) under a Gaussian kernel density of ``samples`` (smooth empirical CDF)."""
    from scipy.special import ndtr

    x = np.asarray(samples, np.float64)
    points = np.asarray(points, np.float64)
    if len(x) == 0:
        return np.zeros_like(points)
    h = kde_bandwidth(x) if bandwidth is None else float(bandwidth)
    return ndtr((points[..., None] - x) / h).mean(axis=-1)


def smoothed_rates(good, defects, thresholds, band: float, costs: CostModel) -> pd.DataFrame:
    """Operating rates for many thresholds from kernel-smoothed score distributions.

    With ~50 held-out good parts an empirical false-reject rate moves in 2% steps and the cost optimum
    snaps to individual samples; the smoothed estimate extrapolates the tails instead.
    """
    t = np.asarray(thresholds, np.float64)
    lo, hi = t - band, t + band
    g_lo, g_hi = kde_cdf(good, lo), kde_cdf(good, hi)
    d_lo, d_hi = kde_cdf(defects, lo), kde_cdf(defects, hi)
    p = costs.defect_prevalence
    frame = pd.DataFrame({
        "threshold": t, "band": float(band), "false_reject_rate": 1 - g_hi, "good_check_rate": g_hi - g_lo,
        "escape_rate": d_lo, "defect_check_rate": d_hi - d_lo, "defect_reject_rate": 1 - d_hi})
    frame["review_rate"] = (1 - p) * frame.good_check_rate + p * frame.defect_check_rate
    frame["false_reject_cost_per_1000"] = 1000 * (1 - p) * frame.false_reject_rate * costs.cost_false_reject
    frame["escape_cost_per_1000"] = 1000 * p * frame.escape_rate * costs.cost_escape
    frame["review_cost_per_1000"] = 1000 * frame.review_rate * costs.cost_review
    frame["expected_cost_per_1000"] = (frame.false_reject_cost_per_1000 + frame.escape_cost_per_1000
                                       + frame.review_cost_per_1000)
    return frame


DEFAULT_BANDS = (0, 2.5, 5, 7.5, 10, 12.5, 15)


def select_threshold_by_cost(scores, labels, costs: CostModel, bands=DEFAULT_BANDS,
                             thresholds=THRESHOLD_GRID, smooth: bool = False) -> dict:
    """Minimise expected cost over (threshold, band) subject to the review-capacity limit.

    ``smooth=True`` evaluates the cost on kernel-smoothed distributions (used for the label-free default
    policy); the returned metrics are always the empirical ones on ``scores``/``labels``.
    """
    scores = np.asarray(scores, np.float64)
    labels = np.asarray(labels).astype(int)
    best = None
    for band in bands:
        if smooth:
            frame = smoothed_rates(scores[labels == 0], scores[labels == 1], thresholds, band, costs)
            frame = frame[frame.review_rate <= costs.review_capacity]
            candidates = [(round(r.expected_cost_per_1000, 6), r.escape_rate, band, r.threshold)
                          for r in frame.itertuples()]
        else:
            candidates = []
            for t in thresholds:
                m = operating_metrics(scores, labels, t, band, costs)
                if m["review_rate"] <= costs.review_capacity:
                    candidates.append((round(m["expected_cost_per_1000"], 6), m["escape_rate"], band, float(t)))
        for key in candidates:
            if best is None or key < best:
                best = key
    if best is None:  # capacity infeasible -> no CHECK band
        return select_threshold_by_cost(scores, labels, costs, bands=(0,), thresholds=thresholds, smooth=smooth)
    return operating_metrics(scores, labels, best[3], best[2], costs)


def calibration_sets(groups: dict, min_auroc: float = 0.9) -> tuple[np.ndarray, list[tuple[str, np.ndarray, float, bool]]]:
    """Label-free calibration defects: every synthetic set the detector can reliably separate from good.

    ``groups`` holds normalised scores: ``cal`` (held-out good), ``syn`` (full-strength synthetic),
    optionally ``sev`` + ``sev_levels`` (reduced-strength replays) and ``diff`` (diffusion-refined).
    A set joins the default threshold only if its AUROC against held-out good is >= ``min_auroc``
    (full-strength synthetic always joins): sets below the detection limit would otherwise drive the
    cost optimum to "reject everything". Returns (defect scores, [(name, scores, auroc, used), ...]).
    """
    from .metrics import safe_auroc

    good = np.asarray(groups["cal"], np.float64)
    candidates = {"synthetic 100%": np.asarray(groups["syn"], np.float64)}
    if groups.get("sev") is not None and groups.get("sev_levels") is not None and len(groups["sev"]):
        sev, levels = np.asarray(groups["sev"], np.float64), np.round(np.asarray(groups["sev_levels"], np.float64), 2)
        for level in sorted(set(levels.tolist()), reverse=True):
            candidates[f"synthetic {int(round(100 * level))}%"] = sev[levels == level]
    if groups.get("diff") is not None and len(groups["diff"]):
        candidates["diffusion-refined"] = np.asarray(groups["diff"], np.float64)
    report, used = [], []
    for name, values in candidates.items():
        auc = safe_auroc(np.r_[np.zeros(len(good)), np.ones(len(values))], np.r_[good, values])
        ok = bool(auc >= min_auroc or name == "synthetic 100%")
        report.append((name, values, float(auc), ok))
        if ok:
            used.append(values)
    return np.concatenate(used), report


def default_policy(groups: dict, costs: CostModel, min_auroc: float = 0.9) -> dict:
    """The deployed label-free operating point: smoothed cost optimum on held-out good + gated synthetic sets."""
    good = np.asarray(groups["cal"], np.float64)
    defects, _ = calibration_sets(groups, min_auroc)
    return select_threshold_by_cost(np.r_[good, defects], np.r_[np.zeros(len(good)), np.ones(len(defects))],
                                    costs, smooth=True)


def select_threshold_good_quantile(good_scores, false_reject_target: float = 0.01, band: float = 5.0) -> float:
    """Label-free policy: REJECT boundary at the (1 - target) quantile of good scores."""
    upper = float(np.quantile(np.asarray(good_scores, np.float64), 1 - false_reject_target))
    return float(np.clip(upper - band, 0, 100))


def reference_costs(costs: CostModel) -> dict:
    """Cost per 1,000 parts of the two status-quo alternatives."""
    p = costs.defect_prevalence
    return {
        "no_inspection_per_1000": 1000 * p * costs.cost_escape,
        "manual_100pct_per_1000": 1000 * (costs.cost_review + p * (1 - costs.human_detection_rate) * costs.cost_escape),
    }
