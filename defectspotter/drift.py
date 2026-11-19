"""Camera / lighting drift: realistic perturbations, acquisition health checks and SPC alarms.

On a real line a dimming lamp, a dirty lens, a knocked camera or a new
supplier's material changes *every* image a little. A good-only detector sees
that as "unusual", so false rejects climb even though parts are fine. We
monitor both the anomaly-score stream (EWMA control chart) and simple image
health signals (brightness, contrast, sharpness) that often move first.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import pandas as pd

# --------------------------------------------------------------- perturbations
DRIFT_KINDS = {
    "lighting_dim": "Lamp ageing / dimmer lighting (exposure x0.45 at full severity)",
    "overexposure": "Brighter lighting or longer exposure (x1.6)",
    "defocus": "Lens out of focus / vibration blur (Gaussian, up to 3% of width)",
    "camera_shift": "Camera bumped: rotation up to 12 deg and 8% shift",
    "white_balance": "Colour temperature change (warmer light)",
    "sensor_noise": "Higher sensor gain / electrical noise",
    "glare": "Stray reflection / uneven illumination hotspot",
}


def apply_drift(image: np.ndarray, kind: str, severity: float, seed: int = 0) -> np.ndarray:
    """Apply one acquisition change at ``severity`` in [0, 1] to an RGB uint8 image."""
    severity = float(np.clip(severity, 0.0, 1.0))
    if severity == 0:
        return image.copy()
    img = image.astype(np.float32)
    h, w = image.shape[:2]
    if kind == "lighting_dim":
        img = img * (1.0 - 0.55 * severity)
    elif kind == "overexposure":
        img = img * (1.0 + 0.6 * severity)
    elif kind == "defocus":
        img = cv2.GaussianBlur(img, (0, 0), max(0.1, 0.03 * w * severity / 3))
    elif kind == "camera_shift":
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), 12 * severity, 1.0)
        matrix[:, 2] += (0.08 * w * severity, 0.05 * h * severity)
        img = cv2.warpAffine(img, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)
    elif kind == "white_balance":
        img = img * np.asarray([1 + 0.25 * severity, 1.0, 1 - 0.25 * severity], np.float32)[None, None]
    elif kind == "sensor_noise":
        img = img + np.random.default_rng(seed).normal(0, 28 * severity, img.shape).astype(np.float32)
    elif kind == "glare":
        yy, xx = np.mgrid[0:h, 0:w]
        cy, cx = 0.35 * h, 0.65 * w
        spot = np.exp(-(((yy - cy) ** 2) + ((xx - cx) ** 2)) / (2 * (0.18 * w) ** 2))
        img = img + (150 * severity * spot)[..., None]
    else:
        raise ValueError(f"Unknown drift kind {kind!r}; choose from {sorted(DRIFT_KINDS)}")
    return np.clip(img, 0, 255).astype(np.uint8)


# -------------------------------------------------------------- health checks
HEALTH_SIGNALS = ("brightness", "contrast", "sharpness")


def reference_from_stats(stats: pd.DataFrame) -> dict:
    """Mean/std of acquisition signals on qualified good images (training set)."""
    return {s: {"mean": float(stats[s].mean()), "std": float(max(stats[s].std(), 1e-6)),
                "p01": float(stats[s].quantile(0.01)), "p99": float(stats[s].quantile(0.99))}
            for s in HEALTH_SIGNALS if s in stats}


def health_check(stats: dict, reference: dict, z_limit: float = 4.0) -> list[dict]:
    """Flag acquisition signals far outside the qualified range (input validation)."""
    flags = []
    for signal, ref in reference.items():
        value = float(stats.get(signal, np.nan))
        if not np.isfinite(value):
            continue
        z = (value - ref["mean"]) / ref["std"]
        flags.append({"signal": signal, "value": value, "reference_mean": ref["mean"], "z": float(z),
                      "status": "OUT OF RANGE" if abs(z) > z_limit else "ok"})
    return flags


# ------------------------------------------------------------------- SPC
@dataclass
class EWMAChart:
    """Exponentially weighted moving average control chart (Roberts 1959)."""

    mean: float
    std: float
    lam: float = 0.2
    width: float = 3.0

    def run(self, values) -> pd.DataFrame:
        values = np.asarray(values, np.float64)
        z = self.mean
        rows = []
        for t, x in enumerate(values, start=1):
            z = self.lam * x + (1 - self.lam) * z
            sigma_z = self.std * np.sqrt(self.lam / (2 - self.lam) * (1 - (1 - self.lam) ** (2 * t)))
            ucl, lcl = self.mean + self.width * sigma_z, self.mean - self.width * sigma_z
            rows.append({"t": t, "value": float(x), "ewma": float(z), "ucl": float(ucl), "lcl": float(lcl),
                         "alarm_high": bool(z > ucl), "alarm_low": bool(z < lcl)})
        return pd.DataFrame(rows)


def population_stability_index(reference, current, bins: int = 10) -> float:
    reference = np.asarray(reference, np.float64)
    current = np.asarray(current, np.float64)
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3 or len(current) == 0:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.clip(np.histogram(reference, edges)[0] / len(reference), 1e-4, None)
    cur = np.clip(np.histogram(current, edges)[0] / len(current), 1e-4, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def first_alarm(frame: pd.DataFrame, start: int, consecutive: int = 2) -> int | None:
    """Index (t) of the first sustained alarm at or after ``start`` (None if never)."""
    alarms = (frame["alarm_high"] | frame["alarm_low"]).to_numpy()
    run = 0
    for i in range(start, len(alarms)):
        run = run + 1 if alarms[i] else 0
        if run >= consecutive:
            return int(frame["t"].iloc[i - consecutive + 1])
    return None
