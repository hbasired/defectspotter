"""Compare candidate PatchCore backbones on accuracy, ONNX size and speed.

The deployment budget asks for a ~10-20 MB ONNX feature extractor, so this
study measures what each backbone buys in image/pixel AUROC for its size.

Example:
    python scripts/backbone_study.py --data /path/to/mvtec --out reports/backbone_study.csv
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectspotter.data import index_mvtec, make_splits, read_mask  # noqa: E402
from defectspotter.features import PatchFeatureExtractor, count_parameters, extract_features  # noqa: E402
from defectspotter.patchcore import MemoryBank, PatchCoreConfig, build_memory_bank, score_grid, topk_mean  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--categories", nargs="+", default=["bottle", "screw", "carpet"])
    parser.add_argument("--configs", nargs="+",
                        default=["mobilenet_v3_small:224", "efficientnet_b0:224", "resnet18:224",
                                 "resnet18:256", "wide_resnet50_2:224"])
    parser.add_argument("--coreset-ratio", type=float, default=0.10)
    parser.add_argument("--out", default="reports/backbone_study.csv")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    index = index_mvtec(args.data, args.categories)
    splits = make_splits(index, seed=args.seed)
    rows = []
    for spec in args.configs:
        backbone, size = spec.split(":")
        size = int(size)
        model = PatchFeatureExtractor(backbone)
        params = count_parameters(model)
        for category in args.categories:
            frame = splits[splits.category == category]
            train_paths = frame[frame.role == "model_train_good"].image_path.tolist()
            test = frame[frame.split == "test"].reset_index(drop=True)
            t0 = time.perf_counter()
            train_grids = extract_features(model, train_paths, size, batch_size=16)
            t_extract = (time.perf_counter() - t0) / len(train_paths)
            t0 = time.perf_counter()
            bank = build_memory_bank(train_grids, coreset_ratio=args.coreset_ratio, seed=args.seed)
            t_coreset = time.perf_counter() - t0
            memory = MemoryBank(bank)
            config = PatchCoreConfig(backbone=backbone, image_size=size)
            test_grids = extract_features(model, test.image_path.tolist(), size, batch_size=16)
            max_scores, topk_scores, maps = [], [], []
            t0 = time.perf_counter()
            for grid in test_grids:
                score, amap, _ = score_grid(grid.astype(np.float32), memory, config)
                max_scores.append(score)
                topk_scores.append(topk_mean(amap, 0.01))
                maps.append(amap)
            t_score = (time.perf_counter() - t0) / len(test_grids)
            labels = test.label.to_numpy()
            masks = np.stack([read_mask(p if p else None, size) for p in test.mask_path])
            pixel_auroc = roc_auc_score(masks.ravel(), np.stack(maps).ravel())
            row = {
                "backbone": backbone, "image_size": size, "category": category,
                "image_auroc_max": roc_auc_score(labels, max_scores),
                "image_auroc_topk": roc_auc_score(labels, topk_scores),
                "pixel_auroc": pixel_auroc,
                "feature_dim": int(train_grids.shape[-1]), "grid": int(train_grids.shape[1]),
                "bank_vectors": len(memory), "bank_mb_fp16": memory.nbytes_fp16 / 1024 ** 2,
                "extractor_params_m": params / 1e6, "onnx_mb_fp32_est": params * 4 / 1024 ** 2,
                "extract_ms_per_image": 1000 * t_extract, "knn_ms_per_image": 1000 * t_score,
                "coreset_seconds": t_coreset,
            }
            rows.append(row)
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}, flush=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print(pd.DataFrame(rows).groupby(["backbone", "image_size"])[
        ["image_auroc_max", "image_auroc_topk", "pixel_auroc", "onnx_mb_fp32_est", "extract_ms_per_image"]].mean())


if __name__ == "__main__":
    main()
