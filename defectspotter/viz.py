"""Heat-map colouring, overlays and suspected-region extraction (OpenCV/NumPy only)."""
from __future__ import annotations

import cv2
import numpy as np
import pandas as pd

REGION_COLUMNS = ["region", "x", "y", "width", "height", "area_pct", "peak", "mean"]

# Single "semantic heat" ramp (light yellow -> orange -> red -> dark red); a rainbow
# map such as JET would invent false boundaries in a magnitude.
_HEAT_ANCHORS = np.array([[255, 247, 188], [254, 196, 79], [236, 112, 20], [204, 40, 40], [120, 10, 30]], np.float32)
HEAT_LUT = np.stack([np.interp(np.linspace(0, 1, 256), np.linspace(0, 1, len(_HEAT_ANCHORS)), _HEAT_ANCHORS[:, c])
                     for c in range(3)], axis=1).astype(np.uint8)
HEAT_COLORSCALE = [[i / (len(_HEAT_ANCHORS) - 1), f"rgb({r:.0f},{g:.0f},{b:.0f})"]
                   for i, (r, g, b) in enumerate(_HEAT_ANCHORS)]


def colorize(map_0_100: np.ndarray) -> np.ndarray:
    """0-100 map -> RGB heat map (values >= 100 saturate)."""
    index = (np.clip(np.asarray(map_0_100, np.float32), 0, 100) * 2.55).astype(np.uint8)
    return HEAT_LUT[index]


def overlay(image: np.ndarray, map_0_100: np.ndarray, floor: float = 30.0, max_alpha: float = 0.8) -> np.ndarray:
    """Blend the heat map over the image; values below ``floor`` stay transparent."""
    h, w = image.shape[:2]
    resized = cv2.resize(np.asarray(map_0_100, np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    heat = colorize(resized)
    alpha = (np.clip((resized - floor) / max(1e-6, 100 - floor), 0, 1) ** 0.7 * max_alpha)[..., None]
    return (image.astype(np.float32) * (1 - alpha) + heat.astype(np.float32) * alpha).astype(np.uint8)


def find_regions(map_0_100: np.ndarray, level: float = 50.0, min_area_pct: float = 0.05,
                 max_regions: int = 6, out_size: tuple[int, int] | None = None) -> pd.DataFrame:
    """Connected components of the normalised map above ``level`` (box coords in out_size)."""
    anomaly = np.asarray(map_0_100, np.float32)
    binary = (anomaly >= level).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    sy = (out_size[1] / anomaly.shape[0]) if out_size else 1.0
    sx = (out_size[0] / anomaly.shape[1]) if out_size else 1.0
    rows = []
    for k in range(1, count):
        x, y, w, h, area = stats[k]
        area_pct = 100.0 * area / binary.size
        if area_pct < min_area_pct:
            continue
        values = anomaly[labels == k]
        rows.append({"x": int(x * sx), "y": int(y * sy), "width": int(max(1, w * sx)), "height": int(max(1, h * sy)),
                     "area_pct": round(float(area_pct), 3), "peak": round(float(values.max()), 1),
                     "mean": round(float(values.mean()), 1)})
    frame = pd.DataFrame(rows, columns=REGION_COLUMNS[1:])
    if frame.empty:
        return pd.DataFrame(columns=REGION_COLUMNS)
    frame = frame.sort_values(["peak", "area_pct"], ascending=False).head(max_regions).reset_index(drop=True)
    frame.insert(0, "region", [f"R{i + 1}" for i in range(len(frame))])
    return frame


def draw_regions(image: np.ndarray, regions: pd.DataFrame, color=(220, 38, 38)) -> np.ndarray:
    out = np.ascontiguousarray(image.copy())
    thickness = max(2, image.shape[0] // 220)
    for _, row in regions.iterrows():
        x, y, w, h = int(row["x"]), int(row["y"]), int(row["width"]), int(row["height"])
        pad = max(3, image.shape[0] // 100)
        cv2.rectangle(out, (max(0, x - pad), max(0, y - pad)),
                      (min(out.shape[1] - 1, x + w + pad), min(out.shape[0] - 1, y + h + pad)), color, thickness)
        label = f"{row['region']}  {float(row['peak']):.0f}"
        scale = max(0.45, image.shape[0] / 900)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
        top = max(0, y - pad - th - 8)
        cv2.rectangle(out, (max(0, x - pad), top), (max(0, x - pad) + tw + 8, top + th + 8), color, -1)
        cv2.putText(out, label, (max(0, x - pad) + 4, top + th + 3), cv2.FONT_HERSHEY_SIMPLEX, scale,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def fit_display(image: np.ndarray, max_side: int = 512) -> np.ndarray:
    h, w = image.shape[:2]
    scale = max_side / max(h, w)
    if scale >= 1:
        return image
    return cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
