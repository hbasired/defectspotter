"""Label-free synthetic defects: cut-paste, scratches and stains with exact masks.

These anomalies let us *choose an operating threshold without any real defect
images*. They are a calibration tool, not proof of real-defect performance -
the pipeline measures how far the synthetic threshold lands from one chosen
with real labelled defects.

NumPy/OpenCV only, so the app can also generate examples live.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance

from .constants import TEXTURE_CATEGORIES
from .data import read_rgb

SYNTHETIC_KINDS = ("cutpaste", "scratch", "stain")


def estimate_foreground(image: np.ndarray, category: str | None = None) -> np.ndarray:
    """Part mask so object defects land on the part, not on the background.

    Colour distance (CIELAB) to the median border colour, thresholded with Otsu's method and cleaned
    morphologically; specks smaller than 10% of the largest component are dropped. (The first version
    thresholded at a fixed percentile, which forced >= 45% of every image to be "foreground": a screw
    covers ~13% of the frame, so most of its synthetic defects landed on the empty tray.)
    """
    h, w = image.shape[:2]
    if category in TEXTURE_CATEGORIES:
        return np.ones((h, w), dtype=np.uint8)
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB).astype(np.float32)
    b = max(2, h // 30)
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    distance = cv2.GaussianBlur(np.linalg.norm(lab - np.median(border, axis=0)[None, None, :], axis=2), (0, 0), 2)
    peak = max(float(distance.max()), 1e-6)
    level, _ = cv2.threshold((distance / peak * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (distance > max(12.0, level / 255 * peak)).astype(np.uint8)
    kernel = np.ones((max(3, h // 80) | 1, max(3, w // 80) | 1), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        mask = np.isin(labels, 1 + np.flatnonzero(areas >= 0.1 * areas.max())).astype(np.uint8)
    return mask if mask.mean() >= 0.01 else np.ones((h, w), dtype=np.uint8)


def _soft(mask: np.ndarray, radius: int) -> np.ndarray:
    radius = max(1, int(radius)) | 1
    return cv2.GaussianBlur(mask.astype(np.float32), (radius, radius), 0).clip(0, 1)


def _center_on(rng: np.random.Generator, foreground: np.ndarray, ph: int, pw: int) -> tuple[int, int]:
    h, w = foreground.shape
    for _ in range(80):
        cy = int(rng.integers(ph // 2, max(ph // 2 + 1, h - ph // 2)))
        cx = int(rng.integers(pw // 2, max(pw // 2 + 1, w - pw // 2)))
        y0, y1 = max(0, cy - ph // 2), min(h, cy + math.ceil(ph / 2))
        x0, x1 = max(0, cx - pw // 2), min(w, cx + math.ceil(pw / 2))
        if foreground[y0:y1, x0:x1].mean() > 0.45:
            return cy, cx
    ys, xs = np.nonzero(foreground)
    if len(xs):
        i = int(rng.integers(len(xs)))
        return int(ys[i]), int(xs[i])
    return h // 2, w // 2


def _cutpaste(image: np.ndarray, fg: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Cut a patch, rotate/recolour it and paste it elsewhere on the part (Li et al. 2021)."""
    h, w = image.shape[:2]
    ph = int(rng.integers(max(6, h // 40), max(8, h // 7)))
    pw = int(rng.integers(max(6, w // 40), max(8, w // 6)))
    sy, sx = _center_on(rng, fg, ph, pw)
    y0, y1 = max(0, sy - ph // 2), min(h, sy + math.ceil(ph / 2))
    x0, x1 = max(0, sx - pw // 2), min(w, sx + math.ceil(pw / 2))
    patch = Image.fromarray(image[y0:y1, x0:x1])
    patch = patch.rotate(float(rng.uniform(-45, 45)), resample=Image.Resampling.BILINEAR, expand=True)
    patch = ImageEnhance.Contrast(patch).enhance(float(rng.uniform(0.6, 1.5)))
    patch = ImageEnhance.Brightness(patch).enhance(float(rng.uniform(0.65, 1.35)))
    arr = np.asarray(patch)
    qh, qw = arr.shape[:2]
    dy, dx = _center_on(rng, fg, qh, qw)
    yy0, yy1 = max(0, dy - qh // 2), min(h, dy - qh // 2 + qh)
    xx0, xx1 = max(0, dx - qw // 2), min(w, dx - qw // 2 + qw)
    arr = arr[: yy1 - yy0, : xx1 - xx0]
    local = _soft(np.ones(arr.shape[:2], np.float32), max(3, min(qh, qw) // 7)) * fg[yy0:yy1, xx0:xx1]
    alpha = local[..., None] * float(rng.uniform(0.6, 0.95))
    out = image.astype(np.float32).copy()
    out[yy0:yy1, xx0:xx1] = (1 - alpha) * out[yy0:yy1, xx0:xx1] + alpha * arr.astype(np.float32)
    mask = np.zeros((h, w), np.float32)
    mask[yy0:yy1, xx0:xx1] = local
    return out.clip(0, 255).astype(np.uint8), (mask > 0.25).astype(np.uint8)


def _scratch(image: np.ndarray, fg: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Thin jittered polyline that is darker or brighter than the surface."""
    h, w = image.shape[:2]
    length = int(rng.integers(max(16, min(h, w) // 10), max(24, min(h, w) // 2)))
    thickness = int(rng.integers(max(1, min(h, w) // 300), max(3, min(h, w) // 90)))
    cy, cx = _center_on(rng, fg, thickness * 8, length)
    angle = float(rng.uniform(0, 2 * np.pi))
    steps = int(rng.integers(3, 7))
    points = []
    for i in range(steps):
        t = (i / max(1, steps - 1) - 0.5) * length
        jitter = float(rng.normal(0, max(1.0, length * 0.04)))
        x = int(cx + t * np.cos(angle) - jitter * np.sin(angle))
        y = int(cy + t * np.sin(angle) + jitter * np.cos(angle))
        points.append((int(np.clip(x, 0, w - 1)), int(np.clip(y, 0, h - 1))))
    mask = np.zeros((h, w), np.uint8)
    cv2.polylines(mask, [np.asarray(points, np.int32)], False, 255, thickness=thickness, lineType=cv2.LINE_AA)
    mask = ((mask > 0) & (fg > 0)).astype(np.uint8)
    base = image.astype(np.float32)
    surface = base[fg > 0].mean(axis=0) if np.any(fg) else base.reshape(-1, 3).mean(axis=0)
    sign = -1.0 if rng.random() < 0.65 else 1.0
    colour = np.clip(surface + sign * rng.uniform(50, 120), 0, 255)
    painted = np.clip(colour[None, None, :] + rng.normal(0, 10, base.shape).astype(np.float32), 0, 255)
    alpha = _soft(mask, thickness * 2 + 1)[..., None] * float(rng.uniform(0.7, 1.0))
    return ((1 - alpha) * base + alpha * painted).astype(np.uint8), mask


def _stain(image: np.ndarray, fg: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Irregular noise-shaped blob with a colour/brightness shift (oil, dirt, discolouration)."""
    h, w = image.shape[:2]
    noise = cv2.GaussianBlur(rng.normal(size=(max(8, h // 24), max(8, w // 24))).astype(np.float32), (0, 0), 1.2)
    field = cv2.resize(noise, (w, h), interpolation=cv2.INTER_CUBIC)
    cy, cx = _center_on(rng, fg, h // 4, w // 4)
    yy, xx = np.ogrid[:h, :w]
    ry, rx = float(rng.uniform(h * 0.04, h * 0.16)), float(rng.uniform(w * 0.04, w * 0.18))
    ellipse = ((yy - cy) ** 2 / (ry ** 2 + 1e-6) + (xx - cx) ** 2 / (rx ** 2 + 1e-6)) <= 1
    threshold = float(np.quantile(field[ellipse], rng.uniform(0.4, 0.65))) if np.any(ellipse) else 0.0
    mask = ((field > threshold) & ellipse & (fg > 0)).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=2)
    base = image.astype(np.float32)
    stain = np.clip(base * rng.uniform(0.45, 1.25) + rng.uniform(-50, 50, size=3)[None, None, :], 0, 255)
    alpha = _soft(mask, max(5, min(h, w) // 45))[..., None] * float(rng.uniform(0.45, 0.85))
    return ((1 - alpha) * base + alpha * stain).astype(np.uint8), mask


def make_synthetic_defect(image: np.ndarray, category: str | None = None, seed: int | None = None,
                          kind: str | None = None) -> tuple[np.ndarray, np.ndarray, str]:
    """Return (defective_image, binary_mask, kind) for a good RGB image."""
    rng = np.random.default_rng(seed)
    fg = estimate_foreground(image, category)
    chosen = kind or str(rng.choice(SYNTHETIC_KINDS, p=[0.4, 0.3, 0.3]))
    generator = {"cutpaste": _cutpaste, "scratch": _scratch, "stain": _stain}.get(chosen)
    if generator is None:
        raise ValueError(f"Unknown synthetic kind {chosen!r}; choose from {SYNTHETIC_KINDS}")
    out, mask = generator(image, fg, rng)
    if mask.sum() < 4:  # degenerate placement: retry deterministically
        return make_synthetic_defect(image, category, (seed or 0) + 7919, "cutpaste")
    return out, mask, chosen


def generate_synthetic_set(good_paths: Sequence[str | Path], category: str, output_dir: str | Path,
                           per_image: int = 1, seed: int = 42, max_side: int | None = None) -> pd.DataFrame:
    """Write synthetic defects + masks for a list of good images and return a manifest."""
    output_dir = Path(output_dir)
    (output_dir / "images").mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    rows = []
    for i, path in enumerate(good_paths):
        source = read_rgb(path)
        if max_side and max(source.shape[:2]) > max_side:
            scale = max_side / max(source.shape[:2])
            source = cv2.resize(source, (int(source.shape[1] * scale), int(source.shape[0] * scale)),
                                interpolation=cv2.INTER_AREA)
        for j in range(per_image):
            sample_seed = seed + i * 1009 + j * 9176
            image, mask, kind = make_synthetic_defect(source, category, sample_seed)
            stem = f"{Path(path).stem}_{j:02d}_{kind}"
            image_path = output_dir / "images" / f"{stem}.png"
            mask_path = output_dir / "masks" / f"{stem}_mask.png"
            Image.fromarray(image).save(image_path)
            Image.fromarray(mask * 255).save(mask_path)
            rows.append({"category": category, "source_path": str(path), "image_path": str(image_path),
                         "mask_path": str(mask_path), "synthetic_type": kind, "seed": sample_seed,
                         "mask_area_pct": float(100 * mask.mean())})
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "manifest.csv", index=False)
    return frame
