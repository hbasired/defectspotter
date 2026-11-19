"""Dataset indexing, image I/O, acquisition statistics and leakage-free splits.

Everything here depends only on NumPy, OpenCV, Pillow and pandas so it can be
used by both the training pipeline and the lightweight inference runtime.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Iterable, Sequence

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from .constants import IMAGE_EXTENSIONS, MVTEC_HF_REPO, MVTEC_HF_REVISION

ImageLike = "np.ndarray | Image.Image | str | Path"


# --------------------------------------------------------------------------- I/O
def read_rgb(path: str | Path) -> np.ndarray:
    """Read any image file as an RGB ``uint8`` array (H, W, 3)."""
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def to_rgb_array(image) -> np.ndarray:
    """Convert a path, PIL image or array (gray/RGB/RGBA) to RGB ``uint8``."""
    if isinstance(image, (str, Path)):
        return read_rgb(image)
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"))
    array = np.asarray(image)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=2)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        if array.max() <= 1.0:
            array = array * 255.0
        array = np.clip(array, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(array)


def read_mask(path: str | Path | None, size: int | tuple[int, int] | None = None) -> np.ndarray:
    """Read a ground-truth mask as a binary ``uint8`` array; missing masks are empty."""
    if path is None or (isinstance(path, float) and np.isnan(path)) or str(path) in {"", "nan", "None"}:
        if size is None:
            raise ValueError("size is required when no mask exists")
        width, height = (size, size) if isinstance(size, int) else size
        return np.zeros((height, width), dtype=np.uint8)
    with Image.open(path) as image:
        mask = np.asarray(image.convert("L"))
    if size is not None:
        width, height = (size, size) if isinstance(size, int) else size
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
    return (mask > 0).astype(np.uint8)


def resize_rgb(image: np.ndarray, size: int | tuple[int, int]) -> np.ndarray:
    """Resize with area interpolation when shrinking and bilinear when enlarging."""
    width, height = (size, size) if isinstance(size, int) else size
    shrinking = image.shape[0] > height or image.shape[1] > width
    interpolation = cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR
    return cv2.resize(image, (width, height), interpolation=interpolation)


def imagenet_input(image, image_size: int) -> np.ndarray:
    """RGB image -> (3,S,S) float32, ImageNet-normalised (shared by training and ONNX runtime)."""
    from .constants import IMAGENET_MEAN, IMAGENET_STD

    array = resize_rgb(to_rgb_array(image), image_size).astype(np.float32) / 255.0
    array = (array - np.asarray(IMAGENET_MEAN, np.float32)) / np.asarray(IMAGENET_STD, np.float32)
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def sha256_file(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def save_json(payload, path: str | Path) -> None:
    """Atomically write JSON (NaN/inf are written as null)."""

    def _clean(value):
        if isinstance(value, dict):
            return {str(k): _clean(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [_clean(v) for v in value]
        if isinstance(value, (np.floating, float)):
            value = float(value)
            return value if np.isfinite(value) else None
        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.ndarray):
            return _clean(value.tolist())
        if isinstance(value, Path):
            return str(value)
        return value

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(_clean(payload), indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def load_json(path: str | Path, default=None):
    path = Path(path)
    if not path.exists():
        return default
    return json.loads(path.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------- dataset
def _image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)


def index_mvtec_category(root: str | Path, category: str) -> pd.DataFrame:
    """Index one MVTec AD category (label 0 = good, 1 = defective)."""
    category_root = Path(root) / category
    if not category_root.exists():
        raise FileNotFoundError(f"Category folder does not exist: {category_root}")
    rows: list[dict] = []
    for path in _image_files(category_root / "train" / "good"):
        rows.append({"category": category, "split": "train", "label": 0, "defect_type": "good",
                     "image_path": str(path), "mask_path": ""})
    test_root = category_root / "test"
    for defect_dir in sorted(p for p in test_root.iterdir() if p.is_dir()):
        defect_type = defect_dir.name
        label = int(defect_type != "good")
        for path in _image_files(defect_dir):
            mask_path = ""
            if label:
                candidate = category_root / "ground_truth" / defect_type / f"{path.stem}_mask.png"
                if candidate.exists():
                    mask_path = str(candidate)
            rows.append({"category": category, "split": "test", "label": label, "defect_type": defect_type,
                         "image_path": str(path), "mask_path": mask_path})
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise RuntimeError(f"No images found under {category_root}")
    return frame.sort_values(["split", "defect_type", "image_path"]).reset_index(drop=True)


def index_mvtec(root: str | Path, categories: Sequence[str]) -> pd.DataFrame:
    return pd.concat([index_mvtec_category(root, c) for c in categories], ignore_index=True)


def download_mvtec(destination: str | Path, categories: Sequence[str],
                   repo_id: str = MVTEC_HF_REPO, revision: str | None = MVTEC_HF_REVISION) -> Path:
    """Download selected MVTec AD categories from the pinned Hugging Face mirror.

    MVTec AD is CC BY-NC-SA 4.0: research and teaching use only.
    """
    from huggingface_hub import snapshot_download

    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    patterns = ["README.md"] + [f"{category}/**" for category in categories]
    snapshot_download(repo_id=repo_id, repo_type="dataset", revision=revision,
                      allow_patterns=patterns, local_dir=str(destination))
    return destination


def make_splits(index: pd.DataFrame, seed: int = 42, calibration_fraction: float = 0.2,
                oracle_fraction: float = 0.35, train_cap: int | None = None) -> pd.DataFrame:
    """Assign every image a role without leaking real defects into training.

    Roles
    -----
    model_train_good      good training images used to fit detectors
    calibration_good      held-out good images: score scaling + synthetic threshold
    oracle_calibration    labelled test subset used *only* to compute the "real-defect"
                          oracle threshold for comparison
    operating_test        remaining test images: final operating metrics
    """
    rng = np.random.default_rng(seed)
    pieces: list[pd.DataFrame] = []
    for category, frame in index.groupby("category", sort=False):
        train = frame[frame.split == "train"].sample(frac=1.0, random_state=seed).reset_index(drop=True)
        if train_cap is not None:
            train = train.head(int(train_cap))
        n_cal = max(4, int(round(len(train) * calibration_fraction)))
        roles = np.array(["model_train_good"] * len(train), dtype=object)
        roles[:n_cal] = "calibration_good"
        train = train.assign(role=roles)

        test = frame[frame.split == "test"].copy()
        test_roles = []
        # Stratify per defect type so every failure mode appears in both test roles.
        for _, group in test.groupby("defect_type", sort=True):
            order = rng.permutation(len(group))
            n_oracle = int(round(len(group) * oracle_fraction))
            if len(group) >= 2:
                n_oracle = min(max(1, n_oracle), len(group) - 1)
            group_roles = np.array(["operating_test"] * len(group), dtype=object)
            group_roles[order[:n_oracle]] = "oracle_calibration"
            test_roles.append(group.assign(role=group_roles))
        pieces.extend([train, *test_roles])
    out = pd.concat(pieces, ignore_index=True)
    return out.sort_values(["category", "role", "defect_type", "image_path"]).reset_index(drop=True)


# -------------------------------------------------------------- acquisition stats
def acquisition_stats(image: np.ndarray) -> dict[str, float]:
    """Camera-health signals: brightness, contrast, sharpness, colour balance."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Measure sharpness at a fixed resolution so values are comparable across cameras.
    small = cv2.resize(gray, (256, 256), interpolation=cv2.INTER_AREA)
    return {
        "brightness": float(gray.mean()),
        "contrast": float(gray.std()),
        "sharpness": float(cv2.Laplacian(small, cv2.CV_32F).var()),
        "r_mean": float(image[..., 0].mean()),
        "g_mean": float(image[..., 1].mean()),
        "b_mean": float(image[..., 2].mean()),
        "saturation_pct": float(100.0 * np.mean(gray >= 250)),
        "dark_pct": float(100.0 * np.mean(gray <= 5)),
    }


def image_statistics(path: str | Path) -> dict[str, float | int]:
    image = read_rgb(path)
    return {"width": int(image.shape[1]), "height": int(image.shape[0]), **acquisition_stats(image)}


def iter_chunks(items: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), size):
        yield items[start:start + size]
