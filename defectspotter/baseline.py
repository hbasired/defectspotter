"""Classical baseline: difference from the mean good image after alignment.

Each image is aligned to a robust reference (median of good parts) with
OpenCV's ECC algorithm, then compared pixel-wise with the per-pixel mean and
standard deviation of all aligned good parts. It is transparent and very fast
for fixtured parts, and it shows exactly where learned features are needed:
randomly rotated parts (screw) and textures (carpet).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .data import resize_rgb, to_rgb_array
from .patchcore import topk_mean

MOTIONS = {"translation": cv2.MOTION_TRANSLATION, "euclidean": cv2.MOTION_EUCLIDEAN,
           "affine": cv2.MOTION_AFFINE}


@dataclass
class MeanBaselineConfig:
    image_size: int = 224
    alignment: str = "euclidean"  # none | translation | euclidean | affine
    pre_blur: float = 1.0
    blur_sigma: float = 3.0
    std_floor: float = 6.0
    topk_fraction: float = 0.01


def align(image: np.ndarray, reference: np.ndarray, mode: str) -> tuple[np.ndarray, float]:
    """ECC-align ``image`` to ``reference``; returns (aligned, correlation or nan)."""
    if mode == "none":
        return image, 1.0
    ref = cv2.cvtColor(reference, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    img = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 60, 1e-5)
    try:
        correlation, warp = cv2.findTransformECC(ref, img, warp, MOTIONS[mode], criteria, None, 5)
        aligned = cv2.warpAffine(image, warp, (reference.shape[1], reference.shape[0]),
                                 flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REFLECT)
        return aligned, float(correlation)
    except cv2.error:
        return image, float("nan")


class MeanImageBaseline:
    def __init__(self, config: MeanBaselineConfig | None = None) -> None:
        self.config = config or MeanBaselineConfig()
        self.reference: np.ndarray | None = None
        self.mean: np.ndarray | None = None
        self.std: np.ndarray | None = None
        self.correlations: list[float] = []

    def _prepare(self, image) -> np.ndarray:
        array = resize_rgb(to_rgb_array(image), self.config.image_size)
        if self.config.pre_blur > 0:
            array = cv2.GaussianBlur(array, (0, 0), self.config.pre_blur)
        return array

    def fit(self, images: Sequence) -> "MeanImageBaseline":
        prepared = [self._prepare(item) for item in images]
        self.reference = np.median(np.stack(prepared[: min(25, len(prepared))]), axis=0).astype(np.uint8)
        aligned, self.correlations = [], []
        for image in prepared:
            out, corr = align(image, self.reference, self.config.alignment)
            aligned.append(out.astype(np.float32))
            self.correlations.append(corr)
        stack = np.stack(aligned)
        self.mean = stack.mean(axis=0)
        self.std = stack.std(axis=0)
        return self

    def predict(self, image) -> tuple[float, np.ndarray]:
        if self.mean is None or self.std is None or self.reference is None:
            raise RuntimeError("Baseline is not fitted")
        aligned, _ = align(self._prepare(image), self.reference, self.config.alignment)
        z = np.abs(aligned.astype(np.float32) - self.mean) / np.maximum(self.std, self.config.std_floor)
        anomaly_map = z.mean(axis=2)
        if self.config.blur_sigma > 0:
            anomaly_map = cv2.GaussianBlur(anomaly_map, (0, 0), self.config.blur_sigma)
        return topk_mean(anomaly_map, self.config.topk_fraction), anomaly_map.astype(np.float32)

    def save(self, folder: str | Path) -> None:
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(folder / "mean_baseline.npz", reference=self.reference,
                            mean=self.mean.astype(np.float16), std=self.std.astype(np.float16),
                            correlations=np.asarray(self.correlations, np.float32))
        (folder / "mean_baseline_config.json").write_text(json.dumps(asdict(self.config), indent=2))

    @classmethod
    def load(cls, folder: str | Path) -> "MeanImageBaseline":
        folder = Path(folder)
        model = cls(MeanBaselineConfig(**json.loads((folder / "mean_baseline_config.json").read_text())))
        arrays = np.load(folder / "mean_baseline.npz")
        model.reference = arrays["reference"]
        model.mean = arrays["mean"].astype(np.float32)
        model.std = arrays["std"].astype(np.float32)
        model.correlations = arrays["correlations"].tolist()
        return model
