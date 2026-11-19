"""Download the MVTec AD categories used by DefectSpotter from a pinned Hugging Face mirror.

MVTec AD is licensed CC BY-NC-SA 4.0 - research and teaching use only, no commercial use.
Official source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad

    python scripts/download_data.py --dest data/mvtec_ad
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectspotter.constants import FEW_SHOT_CATEGORY, MAIN_CATEGORIES, MVTEC_HF_REPO, MVTEC_HF_REVISION  # noqa: E402
from defectspotter.data import download_mvtec, index_mvtec  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dest", default="data/mvtec_ad")
    parser.add_argument("--categories", nargs="+", default=[*MAIN_CATEGORIES, FEW_SHOT_CATEGORY])
    parser.add_argument("--revision", default=MVTEC_HF_REVISION)
    args = parser.parse_args()
    print(f"Downloading {args.categories} from {MVTEC_HF_REPO}@{args.revision[:8]} -> {args.dest}")
    print("Licence: CC BY-NC-SA 4.0 (non-commercial). Please cite Bergmann et al., IJCV 2021 / CVPR 2019.")
    download_mvtec(args.dest, args.categories, revision=args.revision)
    index = index_mvtec(args.dest, args.categories)
    print(index.groupby(["category", "split", "label"]).size().rename("images").to_string())


if __name__ == "__main__":
    main()
