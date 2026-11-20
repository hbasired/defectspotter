"""Pose normalisation for elongated parts (screws): rotate every image to a canonical pose before feature extraction.

CNN features are not rotation-invariant, and screws lie on the tray at any angle, so a memory bank of normal patches
has to cover every orientation of every thread. Classic machine vision fixes the pose first: the part mask
(``synthetic.estimate_foreground``) gives the principal axis (PCA of the mask pixels); the image is rotated about the
part's centroid so the axis is horizontal, the wider end (the screw head) is on the left, and the part is centred.
Rotation keeps a screw's thread handedness (a mirror flip would not). Heat maps are warped back to the original
frame, so overlays and pixel metrics are unchanged. Round parts (bottle top view) have no defined axis and are
never aligned (``elongation`` ~ 1).
"""
from __future__ import annotations

import cv2
import numpy as np

from .synthetic import estimate_foreground

ELONGATION_MIN = 2.0  # principal-axis ratio above which a part is "elongated" (screw ~ 5-7, bottle ~ 1)


def _mask_points(image: np.ndarray, category: str | None, work: int = 256):
    h, w = image.shape[:2]
    scale = work / max(h, w)
    small = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    mask = estimate_foreground(small, category)
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], 1).astype(np.float64) / scale  # full-resolution coordinates


def part_pose(image: np.ndarray, category: str | None = None) -> dict:
    """Centroid, principal-axis angle (degrees, head-left convention) and elongation of the part."""
    pts = _mask_points(image, category)
    h, w = image.shape[:2]
    if len(pts) < 20:
        return {"cx": w / 2, "cy": h / 2, "angle": 0.0, "elongation": 1.0}
    c = pts.mean(0)
    evals, evecs = np.linalg.eigh(np.cov((pts - c).T))
    axis = evecs[:, 1]
    proj = (pts - c) @ axis
    perp = (pts - c) @ np.array([-axis[1], axis[0]])
    lo, hi = proj.min(), proj.max()
    span = max(hi - lo, 1e-6)
    width_lo = np.ptp(perp[proj < lo + 0.15 * span]) if np.any(proj < lo + 0.15 * span) else 0.0
    width_hi = np.ptp(perp[proj > hi - 0.15 * span]) if np.any(proj > hi - 0.15 * span) else 0.0
    angle = float(np.degrees(np.arctan2(axis[1], axis[0])) + (180.0 if width_hi > width_lo else 0.0))
    return {"cx": float(c[0]), "cy": float(c[1]), "angle": angle,
            "elongation": float(np.sqrt(max(evals[1], 1e-9) / max(evals[0], 1e-9)))}


def align_matrix(image: np.ndarray, category: str | None = None) -> np.ndarray:
    """2x3 affine (original -> canonical pose) at full resolution."""
    pose = part_pose(image, category)
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((pose["cx"], pose["cy"]), pose["angle"], 1.0)
    matrix[0, 2] += w / 2 - pose["cx"]
    matrix[1, 2] += h / 2 - pose["cy"]
    return matrix


def background_colour(image: np.ndarray) -> tuple:
    b = max(2, image.shape[0] // 30)
    border = np.concatenate([image[:b].reshape(-1, 3), image[-b:].reshape(-1, 3),
                             image[:, :b].reshape(-1, 3), image[:, -b:].reshape(-1, 3)])
    return tuple(int(v) for v in np.median(border, axis=0))


def align_image(image: np.ndarray, category: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Rotate/translate the part to the canonical pose; the uncovered corners get the background colour."""
    matrix = align_matrix(image, category)
    h, w = image.shape[:2]
    aligned = cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                             borderValue=background_colour(image))
    return aligned, matrix


def unalign_map(aligned_map: np.ndarray, matrix: np.ndarray, image_shape: tuple, out_size: int) -> np.ndarray:
    """Warp a square map computed on the aligned image back to the original frame (``out_size`` x ``out_size``)."""
    h, w = image_shape[:2]
    size = aligned_map.shape[0]
    to_map = np.diag([size / w, size / h, 1.0])
    to_out = np.diag([out_size / w, out_size / h, 1.0])
    full = np.vstack([matrix, [0, 0, 1]])
    # original(out) -> aligned(map):  to_map @ full @ inv(to_out)
    m = (to_map @ full @ np.linalg.inv(to_out))[:2]
    fill = float(np.min(aligned_map))
    return cv2.warpAffine(np.asarray(aligned_map, np.float32), m, (out_size, out_size),
                          flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_CONSTANT,
                          borderValue=fill).astype(np.float32)


def align_mask(mask: np.ndarray, matrix: np.ndarray, out_size: int) -> np.ndarray:
    """Forward-warp a full-resolution binary mask to the aligned frame at ``out_size``."""
    h, w = mask.shape[:2]
    warped = cv2.warpAffine(mask.astype(np.uint8), matrix, (w, h), flags=cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return cv2.resize(warped, (out_size, out_size), interpolation=cv2.INTER_NEAREST)


def is_elongated(images, category: str | None = None) -> bool:
    values = [part_pose(np.asarray(img), category)["elongation"] for img in images]
    return bool(len(values) and np.median(values) >= ELONGATION_MIN)
