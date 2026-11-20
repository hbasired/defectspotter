"""Generative synthetic defects: Stable-Diffusion (SDEdit) refinement of procedural defects on GOOD parts.

Used only to choose / validate the operating threshold without real defects (pipeline stage ``diffusion``).

What we learned: plain inpainting with SD's 9-channel *inpainting* model fills the hole from context, so on a good
part it re-draws a good surface - even when started from a visible defect (mean change inside the mask 2-6 grey
levels, synthetic AUROC 0.4-0.6). We therefore use SDEdit with the *base* SD-1.5 model: a crop around a procedural
defect (cut-paste / scratch / stain, guaranteed visible) is partially noised (``strength`` 0.45) and re-rendered with
a product-specific defect prompt (LCM-LoRA: a handful of steps, CPU-feasible). The defect keeps its structure and
gains realistic texture and shading; only the masked region is composited back.

    pip install -r requirements-diffusion.txt
    python scripts/generate_diffusion_defects.py --per-category 8
    python scripts/train.py --data data/mvtec_ad --stages diffusion calibrate evaluate drift summary

Model licences: stable-diffusion-v1-5/stable-diffusion-v1-5 (CreativeML OpenRAIL-M),
latent-consistency/lcm-lora-sdv1-5 (OpenRAIL++).
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectspotter.data import read_rgb  # noqa: E402

PROMPTS = {
    "bottle": {
        "scratch": "a crack and a chipped broken edge on the rim of a glass bottle, top view, industrial inspection photo",
        "stain": "dirty brown debris contamination inside a glass bottle, top view, industrial inspection photo",
        "cutpaste": "a broken glass fragment lying on the rim of a glass bottle, top view, industrial inspection photo",
    },
    "screw": {
        "scratch": "a deep scratch and a dented damaged thread on a steel screw, macro inspection photo",
        "stain": "a rust stain and corrosion spot on a steel screw, macro inspection photo",
        "cutpaste": "a deformed damaged thread on a steel screw, macro inspection photo",
    },
    "carpet": {
        "scratch": "a torn cut and a loose pulled thread in woven carpet fabric, close-up photo",
        "stain": "a dark wine stain soaked into carpet fabric, close-up photo",
        "cutpaste": "a misplaced patch of different carpet fabric, close-up photo",
    },
}
NEGATIVE = "cartoon, drawing, text, watermark, blurry, low quality"
MODEL = "stable-diffusion-v1-5/stable-diffusion-v1-5"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


def load_pipeline(device: str):
    import torch
    from diffusers import AutoPipelineForImage2Image, LCMScheduler

    dtype = torch.float16 if device == "cuda" else torch.float32
    # fp16 *safetensors* are loaded (no pickled checkpoints) and cast to fp32 on CPU.
    pipe = AutoPipelineForImage2Image.from_pretrained(MODEL, variant="fp16", torch_dtype=dtype, use_safetensors=True,
                                                      safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA)
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def crop_box(mask: np.ndarray, min_side: int) -> tuple[int, int, int, int]:
    """Square crop around the defect with generous context (SD needs surroundings to shade realistically)."""
    ys, xs = np.nonzero(mask)
    h, w = mask.shape
    cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
    side = int(max(min_side, 2.2 * max(ys.max() - ys.min(), xs.max() - xs.min())))
    side = min(side, h, w)
    y0 = int(np.clip(cy - side // 2, 0, h - side))
    x0 = int(np.clip(cx - side // 2, 0, w - side))
    return y0, y0 + side, x0, x0 + side


def soften(mask: np.ndarray, image_height: int) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(1.0, image_height / 300))[..., None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--work-dir", default=str(ROOT.parent / "defectspotter_work"))
    parser.add_argument("--categories", nargs="+", default=list(PROMPTS))
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--strength", type=float, default=0.45, help="SDEdit strength (0 = keep the defect, 1 = repaint)")
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if _cuda() else "cpu")
    args = parser.parse_args()

    import torch

    pipe = load_pipeline(args.device)
    gallery = ROOT / "examples" / "diffusion"
    gallery.mkdir(parents=True, exist_ok=True)
    for category in args.categories:  # only this run's products: the script can run one product at a time
        for old_file in gallery.glob(f"{category}_*.jpg"):
            old_file.unlink()
    for category in args.categories:
        out = Path(args.work_dir) / "diffusion" / category
        (out / "images").mkdir(parents=True, exist_ok=True)
        (out / "masks").mkdir(parents=True, exist_ok=True)
        procedural = pd.read_csv(Path(args.work_dir) / "synthetic" / category / "manifest.csv")
        # balanced, seeded pick across the three procedural kinds
        quota = max(1, args.per_category // 3 + 1)
        groups = [group.sample(n=min(len(group), quota), random_state=args.seed)
                  for _, group in procedural.groupby("synthetic_type")]
        picks = pd.concat(groups).head(args.per_category).reset_index(drop=True)
        rows = []
        for k, row in picks.iterrows():
            kind = row.synthetic_type
            source = read_rgb(row.source_path)
            defect = read_rgb(row.image_path)
            mask = (cv2.imread(row.mask_path, cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
            h, w = mask.shape
            context = cv2.dilate(mask, np.ones((max(3, h // 80) | 1, max(3, w // 80) | 1), np.uint8))
            size = args.resolution
            y0, y1, x0, x1 = crop_box(context, min_side=min(h, w) // 4)
            crop = cv2.resize(defect[y0:y1, x0:x1], (size, size), interpolation=cv2.INTER_AREA)
            started = time.time()
            refined = pipe(prompt=PROMPTS[category][kind], negative_prompt=NEGATIVE, image=Image.fromarray(crop),
                           num_inference_steps=args.steps, strength=args.strength, guidance_scale=args.guidance,
                           generator=torch.Generator("cpu").manual_seed(args.seed + k)).images[0]
            painted = defect.copy()
            painted[y0:y1, x0:x1] = cv2.resize(np.asarray(refined), (x1 - x0, y1 - y0), interpolation=cv2.INTER_CUBIC)
            soft = soften(context, h)
            result = (source * (1 - soft) + painted * soft).clip(0, 255).astype(np.uint8)
            stem = f"{Path(row.source_path).stem}_diffusion_{kind}_{k:02d}"
            Image.fromarray(result).save(out / "images" / f"{stem}.png")
            Image.fromarray(context * 255).save(out / "masks" / f"{stem}_mask.png")
            change = float(np.abs(result.astype(int) - source.astype(int)).mean(axis=2)[context > 0].mean())
            rows.append({"category": category, "source_path": row.source_path, "image_path": str(out / "images" / f"{stem}.png"),
                         "mask_path": str(out / "masks" / f"{stem}_mask.png"), "synthetic_type": f"diffusion_{kind}",
                         "prompt": PROMPTS[category][kind], "mask_area_pct": float(100 * context.mean()),
                         "mean_change_in_mask": round(change, 1), "seconds": round(time.time() - started, 1)})
            if k < 4:
                Image.fromarray(cv2.resize(result, (320, 320), interpolation=cv2.INTER_AREA)).save(
                    gallery / f"{category}_diffusion_{kind}_{k}.jpg", quality=92)
            print(f"{category} {k + 1}/{len(picks)} {kind}: {rows[-1]['seconds']}s, mean change in mask {change:.1f}", flush=True)
        pd.DataFrame(rows).to_csv(out / "manifest.csv", index=False)
    manifests = sorted((Path(args.work_dir) / "diffusion").glob("*/manifest.csv"))  # every product generated so far
    summary = pd.concat([pd.read_csv(m) for m in manifests])
    summary[["category", "synthetic_type", "prompt", "mask_area_pct", "mean_change_in_mask", "seconds"]].to_csv(
        ROOT / "reports" / "diffusion_summary.csv", index=False)


def _cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


if __name__ == "__main__":
    main()
