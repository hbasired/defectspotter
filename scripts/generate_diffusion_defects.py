"""Generative synthetic defects: Stable-Diffusion refinement of procedural defects on GOOD parts.

Used only to choose / validate the operating threshold without real defects (pipeline stage ``diffusion``).

Why refinement and not plain inpainting: an inpainting model is trained to fill a hole with *plausible* content,
so on a good part it simply re-draws a good surface (we measured mean changes of 2-6 grey levels inside the
mask - invisible). Instead we start from a procedural defect (cut-paste / scratch / stain, guaranteed visible),
noise the masked region partially (``strength`` 0.6, SDEdit) and let SD-1.5 inpainting + LCM-LoRA re-render it
with a defect prompt: the defect keeps its structure but gains realistic texture and shading. Only the masked
region is composited back, so the rest of the part stays pixel-identical.

    pip install -r requirements-diffusion.txt
    python scripts/generate_diffusion_defects.py --data data/mvtec_ad --per-category 8
    python scripts/train.py --data data/mvtec_ad --stages diffusion calibrate evaluate drift summary

Model licences: stable-diffusion-v1-5/stable-diffusion-inpainting (CreativeML OpenRAIL-M),
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
MODEL = "stable-diffusion-v1-5/stable-diffusion-inpainting"
LCM_LORA = "latent-consistency/lcm-lora-sdv1-5"


def load_pipeline(device: str):
    import torch
    from diffusers import AutoPipelineForInpainting, LCMScheduler

    dtype = torch.float16 if device == "cuda" else torch.float32
    # fp16 *safetensors* are loaded (no pickled checkpoints) and cast to fp32 on CPU.
    pipe = AutoPipelineForInpainting.from_pretrained(MODEL, variant="fp16", torch_dtype=dtype, use_safetensors=True,
                                                     safety_checker=None, requires_safety_checker=False)
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(LCM_LORA)
    pipe.fuse_lora()
    pipe.set_progress_bar_config(disable=True)
    return pipe.to(device)


def soften(mask: np.ndarray, image_height: int) -> np.ndarray:
    return cv2.GaussianBlur(mask.astype(np.float32), (0, 0), max(1.0, image_height / 300))[..., None]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True)
    parser.add_argument("--work-dir", default=str(ROOT.parent / "defectspotter_work"))
    parser.add_argument("--categories", nargs="+", default=list(PROMPTS))
    parser.add_argument("--per-category", type=int, default=8)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--strength", type=float, default=0.6, help="SDEdit strength inside the mask (0=keep, 1=repaint)")
    parser.add_argument("--guidance", type=float, default=1.5)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cuda" if _cuda() else "cpu")
    args = parser.parse_args()

    import torch

    pipe = load_pipeline(args.device)
    gallery = ROOT / "examples" / "diffusion"
    if gallery.exists():
        for old_file in gallery.glob("*.jpg"):
            old_file.unlink()
    gallery.mkdir(parents=True, exist_ok=True)
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
            started = time.time()
            refined = pipe(prompt=PROMPTS[category][kind], negative_prompt=NEGATIVE,
                           image=Image.fromarray(cv2.resize(defect, (size, size), interpolation=cv2.INTER_AREA)),
                           mask_image=Image.fromarray(cv2.resize(context * 255, (size, size), interpolation=cv2.INTER_NEAREST)),
                           num_inference_steps=args.steps, strength=args.strength, guidance_scale=args.guidance,
                           generator=torch.Generator("cpu").manual_seed(args.seed + k)).images[0]
            refined = cv2.resize(np.asarray(refined), (w, h), interpolation=cv2.INTER_CUBIC)
            soft = soften(context, h)
            result = (source * (1 - soft) + refined * soft).clip(0, 255).astype(np.uint8)
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
    summary = pd.concat([pd.read_csv(Path(args.work_dir) / "diffusion" / c / "manifest.csv") for c in args.categories])
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
