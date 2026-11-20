"""Run the complete DefectSpotter experiment (CPU is fine; a GPU makes it ~5x faster).

    # 1. download the three product lines + the few-shot product (~1.3 GB, non-commercial licence)
    python scripts/download_data.py --dest data/mvtec_ad
    # 2. train, calibrate, evaluate and export everything the app needs
    python scripts/train.py --data data/mvtec_ad

Stages can be re-run individually, e.g. ``--stages calibrate evaluate summary``.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectspotter.decision import CostModel  # noqa: E402
from defectspotter.pipeline import Pipeline, PipelineConfig  # noqa: E402


def main() -> None:
    defaults = PipelineConfig(data_root="")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="MVTec AD root containing bottle/, screw/, carpet/ ...")
    parser.add_argument("--work-dir", default=str(ROOT.parent / "defectspotter_work"),
                        help="cache folder for features, synthetic images and scores (outside the repo)")
    parser.add_argument("--categories", nargs="+", default=list(defaults.categories))
    parser.add_argument("--few-shot-category", default=defaults.few_shot_category)
    parser.add_argument("--backbone", default=defaults.backbone)
    parser.add_argument("--image-size", type=int, default=defaults.image_size,
                        help="default resolution (few-shot onboarding, ONNX export example)")
    parser.add_argument("--size-grid", type=int, nargs="+", default=list(defaults.size_grid),
                        help="PatchCore input resolutions searched per product (label-free selection)")
    parser.add_argument("--pool-grid", type=int, nargs="+", default=list(defaults.pool_grid),
                        help="feature aggregation sizes searched (3 = PatchCore 3x3 average, 1 = raw descriptors)")
    parser.add_argument("--object-rotations", type=int, nargs="+", default=list(defaults.object_rotation_grid),
                        help="numbers of 90-degree rotations of good parts in the bank searched for objects")
    parser.add_argument("--vae-epochs", type=int, default=defaults.vae_epochs)
    parser.add_argument("--vae-latent-dim", type=int, default=defaults.vae_latent_dim)
    parser.add_argument("--bootstrap", type=int, default=defaults.bootstrap)
    parser.add_argument("--device", default="cuda" if _cuda() else "cpu")
    parser.add_argument("--stages", nargs="+", default=list(defaults.stages))
    parser.add_argument("--quick", action="store_true", help="smoke run: tiny grids, 5 VAE epochs")
    parser.add_argument("--prevalence", type=float, default=defaults.costs.defect_prevalence)
    parser.add_argument("--cost-escape", type=float, default=defaults.costs.cost_escape)
    parser.add_argument("--cost-false-reject", type=float, default=defaults.costs.cost_false_reject)
    parser.add_argument("--cost-review", type=float, default=defaults.costs.cost_review)
    args = parser.parse_args()

    config = PipelineConfig(
        data_root=args.data, project_root=str(ROOT), work_dir=args.work_dir, categories=tuple(args.categories),
        few_shot_category=args.few_shot_category or None, backbone=args.backbone, image_size=args.image_size,
        vae_epochs=5 if args.quick else args.vae_epochs, vae_latent_dim=args.vae_latent_dim,
        bootstrap=100 if args.quick else args.bootstrap, device=args.device, stages=tuple(args.stages),
        costs=CostModel(defect_prevalence=args.prevalence, cost_escape=args.cost_escape,
                        cost_false_reject=args.cost_false_reject, cost_review=args.cost_review),
        size_grid=tuple(args.size_grid), pool_grid=tuple(args.pool_grid),
        object_rotation_grid=tuple(args.object_rotations),
    )
    if args.quick:
        config.coreset_grid = (0.05,)
        config.blur_grid = (4.0,)
        config.score_modes = ("max",)
        config.size_grid = (args.image_size,)
        config.object_rotation_grid = (1,)
    Pipeline(config).run()


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
