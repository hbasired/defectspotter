"""PatchCore-style memory bank: greedy coreset selection and nearest-neighbour scoring.

Training builds a memory bank of patch descriptors from good images only and
compresses it with greedy k-center (minimax facility location) coreset
selection, which keeps rare-but-normal patches that random subsampling would
drop. At inference every test patch is scored by its distance to the closest
normal patch; the image score is the most anomalous patch.

Scoring is pure NumPy so the deployed app needs only ONNX Runtime.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass
class PatchCoreConfig:
    backbone: str = "resnet18"
    image_size: int = 224
    coreset_ratio: float = 0.10
    max_bank_mb: float = 45.0
    projection_dim: int = 128
    blur_sigma: float = 4.0
    score_mode: str = "max"  # "max" of patch distances or "topk" mean of the smoothed map
    topk_fraction: float = 0.01
    pool_size: int = 3  # local neighbourhood average of the CNN features (3 = PatchCore default, 1 = raw)
    bank_rotations: int = 1  # good images were added to the bank at this many 90-degree rotations (train only)

    def to_dict(self) -> dict:
        return asdict(self)


def greedy_coreset(features: np.ndarray, n_select: int, projection_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Return indices of a greedy k-center coreset (Sener & Savarese 2018; PatchCore 2022).

    A Johnson-Lindenstrauss random projection makes each iteration cheap; the
    selection is then applied to the full-dimensional features.
    """
    features = np.asarray(features, dtype=np.float32)
    n_total, dim = features.shape
    if n_select >= n_total:
        return np.arange(n_total, dtype=np.int64)
    rng = np.random.default_rng(seed)
    try:
        import torch

        torch_features = torch.from_numpy(features)
        if projection_dim and projection_dim < dim:
            projection = torch.from_numpy(
                rng.standard_normal((dim, projection_dim)).astype(np.float32) / math.sqrt(projection_dim))
            z = torch_features @ projection
        else:
            z = torch_features.clone()
        z_norm = (z * z).sum(1)
        start = int(rng.integers(n_total))
        selected = np.empty(n_select, dtype=np.int64)
        selected[0] = start
        center = z[start]
        min_dist = z_norm - 2.0 * (z @ center) + center.dot(center)
        for i in range(1, n_select):
            index = int(torch.argmax(min_dist))
            selected[i] = index
            center = z[index]
            torch.minimum(min_dist, z_norm - 2.0 * (z @ center) + center.dot(center), out=min_dist)
        return selected
    except ImportError:  # NumPy fallback (slower) for torch-free environments
        if projection_dim and projection_dim < dim:
            projection = rng.standard_normal((dim, projection_dim)).astype(np.float32) / math.sqrt(projection_dim)
            z = features @ projection
        else:
            z = features.copy()
        z_norm = np.einsum("ij,ij->i", z, z)
        selected = np.empty(n_select, dtype=np.int64)
        selected[0] = int(rng.integers(n_total))
        center = z[selected[0]]
        min_dist = z_norm - 2.0 * (z @ center) + center @ center
        for i in range(1, n_select):
            index = int(np.argmax(min_dist))
            selected[i] = index
            center = z[index]
            np.minimum(min_dist, z_norm - 2.0 * (z @ center) + center @ center, out=min_dist)
        return selected


def coreset_size(n_vectors: int, dim: int, coreset_ratio: float, max_bank_mb: float = 45.0) -> int:
    max_vectors = max(1, int(max_bank_mb * 1024 ** 2 // (dim * 2)))  # float16 on disk
    return int(min(n_vectors, max_vectors, max(1, round(n_vectors * coreset_ratio))))


class CoresetLadder:
    """One greedy k-center run serves every coreset ratio.

    Greedy selection is incremental: its first k picks are exactly the size-k greedy coreset, so banks for
    ratios 1% / 5% / 10% are prefixes of a single run at the largest ratio (3 runs -> 1 in model selection).
    Very large patch sets (e.g. good parts x 4 rotations) are first reduced to ``max_candidates`` random
    patches: rotated views are highly redundant, and greedy cost grows with candidates x selections.
    Bank sizes still follow the ratio of the *full* set (and the size cap).
    """

    def __init__(self, patch_grids: np.ndarray, max_ratio: float, max_bank_mb: float = 45.0,
                 projection_dim: int = 128, seed: int = 0, max_candidates: int = 300_000) -> None:
        flat = np.asarray(patch_grids).reshape(-1, patch_grids.shape[-1])
        self.n_total = len(flat)
        if max_candidates and len(flat) > max_candidates:
            keep = np.sort(np.random.default_rng(seed).choice(len(flat), max_candidates, replace=False))
            flat = flat[keep]
        self.flat = flat
        self.max_bank_mb = max_bank_mb
        n_max = min(len(flat), coreset_size(self.n_total, flat.shape[1], max_ratio, max_bank_mb))
        self.order = greedy_coreset(flat.astype(np.float32, copy=False), n_max, projection_dim, seed)

    def bank(self, coreset_ratio: float) -> np.ndarray:
        n = min(len(self.order), coreset_size(self.n_total, self.flat.shape[1], coreset_ratio, self.max_bank_mb))
        return np.ascontiguousarray(self.flat[self.order[:n]].astype(np.float16))


def build_memory_bank(patch_grids: np.ndarray, coreset_ratio: float = 0.1, max_bank_mb: float = 45.0,
                      projection_dim: int = 128, seed: int = 0) -> np.ndarray:
    """Flatten (N,h,w,C) good-image patch grids and keep a float16 greedy coreset."""
    flat = np.asarray(patch_grids, dtype=np.float32).reshape(-1, patch_grids.shape[-1])
    bytes_per_vector = flat.shape[1] * 2  # float16 on disk
    max_vectors = max(1, int(max_bank_mb * 1024 ** 2 // bytes_per_vector))
    n_select = int(min(len(flat), max_vectors, max(1, round(len(flat) * coreset_ratio))))
    indices = greedy_coreset(flat, n_select, projection_dim=projection_dim, seed=seed)
    return np.ascontiguousarray(flat[indices].astype(np.float16))


class MemoryBank:
    """Exact 1-nearest-neighbour search with chunked squared-distance expansion."""

    def __init__(self, bank: np.ndarray, chunk_size: int = 8192) -> None:
        self.bank = np.ascontiguousarray(np.asarray(bank, dtype=np.float32))
        self.bank_sq = np.einsum("ij,ij->i", self.bank, self.bank)
        self.chunk_size = int(chunk_size)

    def __len__(self) -> int:
        return int(self.bank.shape[0])

    @property
    def nbytes_fp16(self) -> int:
        return int(self.bank.size * 2)

    def nearest(self, queries: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (distance, index) of the nearest memory-bank entry for every query."""
        queries = np.ascontiguousarray(np.asarray(queries, dtype=np.float32))
        q_sq = np.einsum("ij,ij->i", queries, queries)[:, None]
        best = np.full(len(queries), np.inf, dtype=np.float32)
        best_index = np.zeros(len(queries), dtype=np.int64)
        for start in range(0, len(self.bank), self.chunk_size):
            chunk = self.bank[start:start + self.chunk_size]
            dist = q_sq - 2.0 * (queries @ chunk.T) + self.bank_sq[None, start:start + self.chunk_size]
            local = np.argmin(dist, axis=1)
            values = dist[np.arange(len(queries)), local]
            better = values < best
            best[better] = values[better]
            best_index[better] = local[better] + start
        return np.sqrt(np.maximum(best, 0.0)), best_index


def anomaly_map_from_patches(patch_scores: np.ndarray, output_size: int, blur_sigma: float = 4.0) -> np.ndarray:
    """Upsample a (h,w) patch-score grid to (S,S) and smooth it (PatchCore uses sigma=4)."""
    score_map = cv2.resize(patch_scores.astype(np.float32), (output_size, output_size),
                           interpolation=cv2.INTER_LINEAR)
    if blur_sigma > 0:
        score_map = cv2.GaussianBlur(score_map, (0, 0), sigmaX=blur_sigma, sigmaY=blur_sigma)
    return score_map.astype(np.float32)


def topk_mean(values: np.ndarray, fraction: float) -> float:
    flat = np.asarray(values, dtype=np.float32).ravel()
    k = max(1, int(math.ceil(len(flat) * fraction)))
    if k >= len(flat):
        return float(flat.mean())
    return float(np.partition(flat, len(flat) - k)[-k:].mean())


def score_patches(patch_scores: np.ndarray, config: PatchCoreConfig) -> tuple[float, np.ndarray]:
    """(h,w) nearest-neighbour distances -> (image_score, anomaly_map SxS) for one image."""
    anomaly_map = anomaly_map_from_patches(patch_scores, config.image_size, config.blur_sigma)
    if config.score_mode == "topk":
        image_score = topk_mean(anomaly_map, config.topk_fraction)
    else:
        image_score = float(patch_scores.max())
    return image_score, anomaly_map


def score_grid(grid: np.ndarray, bank: MemoryBank, config: PatchCoreConfig) -> tuple[float, np.ndarray, np.ndarray]:
    """Score one (h,w,C) patch grid. Returns (image_score, anomaly_map SxS, patch_scores hxw)."""
    h, w, c = grid.shape
    distances, _ = bank.nearest(grid.reshape(-1, c))
    patch_scores = distances.reshape(h, w)
    image_score, anomaly_map = score_patches(patch_scores, config)
    return image_score, anomaly_map, patch_scores


def patch_distances(grids: np.ndarray, bank: MemoryBank, batch: int = 8) -> np.ndarray:
    """Nearest-neighbour distance of every patch: (N,h,w,C) -> (N,h,w). The expensive step, done once."""
    grids = np.asarray(grids)
    n, h, w, c = grids.shape
    out = np.empty((n, h, w), np.float32)
    for start in range(0, n, batch):
        chunk = np.asarray(grids[start:start + batch], np.float32).reshape(-1, c)
        out[start:start + batch] = bank.nearest(chunk)[0].reshape(-1, h, w)
    return out


def scores_from_distances(distances: np.ndarray, config: PatchCoreConfig,
                          map_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Apply a scoring config to precomputed patch distances; optionally resize maps to ``map_size``."""
    scores, maps = [], []
    for patch_scores in distances:
        score, anomaly_map = score_patches(patch_scores, config)
        if map_size and anomaly_map.shape[0] != map_size:
            anomaly_map = cv2.resize(anomaly_map, (map_size, map_size), interpolation=cv2.INTER_LINEAR)
        scores.append(score)
        maps.append(anomaly_map)
    return np.asarray(scores, dtype=np.float32), np.stack(maps).astype(np.float32)


def score_grids(grids: np.ndarray, bank: MemoryBank, config: PatchCoreConfig,
                map_size: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    return scores_from_distances(patch_distances(grids, bank), config, map_size)
