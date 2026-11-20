"""Deploy DefectSpotter (app + trained artifacts + evidence) to a Hugging Face Gradio Space.

    export HF_TOKEN=hf_xxx            # write token; never commit it (or run `huggingface-cli login`)
    python scripts/deploy_hf.py --space bhkr0901/defectspotter

Hardware (Hugging Face rules as of 2026):
  * Free personal accounts (verified e-mail, account > 30 days old) may host up to 2 Gradio Spaces on
    **ZeroGPU** for free -> default ``--hardware zero-a10g``. DefectSpotter runs inference on the CPU of the
    ZeroGPU container with ONNX Runtime, so the daily GPU quota is never consumed by visitors.
  * ``cpu-basic`` Gradio Spaces require a PRO / Team plan to create.
The script prints the Space URL (https://huggingface.co/spaces/<id>) and the direct app URL.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ["app.py", "defectspotter", "ui", "artifacts", "reports", "examples", "docs", "LICENSE"]
SPACE_REQUIREMENTS = """numpy>=1.26
opencv-python-headless>=4.9
pillow>=10.0
pandas>=2.1
scikit-learn>=1.4
onnxruntime>=1.18
plotly>=5.24
huggingface-hub>=0.25
spaces
torch==2.13.0  # ZeroGPU-supported version (only used by the GPU self-test)
"""


def space_card(space_id: str, gradio_version: str) -> str:
    return f"""---
title: DefectSpotter
emoji: 🔎
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {gradio_version}
python_version: "3.12"
app_file: app.py
pinned: true
license: cc-by-nc-sa-4.0
short_description: Visual inspection without defect labels (PatchCore + VAE)
tags:
  - anomaly-detection
  - industrial-inspection
  - patchcore
  - vae
  - onnx
  - mvtec-ad
---

# 🔎 DefectSpotter — visual inspection without defect labels

Upload a part image (bottle, screw or carpet) and get an anomaly score, a heat map of the suspected defect and a
**PASS / CHECK / REJECT** decision. The detectors were trained on **good parts only**: a PatchCore-style memory bank on
frozen ResNet-18 features (ONNX), a feature-space VAE and a pixel VAE (GenAI) and an aligned mean-image baseline. Thresholds are chosen with
synthetic defects and an explicit cost model; CHECK goes to a human review queue; camera/lighting drift is monitored.

* Source code, notebook and full report: see the GitHub repository linked in the app.
* Data: [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad) — CC BY-NC-SA 4.0, research and teaching only.
* The app is a presentation that ends in the live demo (last tab); the 📖 Guide tab explains every model and concept.
* API: `/inspect_part`, `/inspect_batch`, `/health` (see "Use via API").

Space: `{space_id}`
"""


def stage(folder: Path, space_id: str, gradio_version: str) -> None:
    for name in INCLUDE:
        source = ROOT / name
        if not source.exists():
            if name == "LICENSE":
                continue
            raise SystemExit(f"Missing {source}. Train first: python scripts/train.py --data <mvtec>")
        target = folder / name
        if source.is_dir():
            shutil.copytree(source, target, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
        else:
            shutil.copy2(source, target)
    (folder / "requirements.txt").write_text(SPACE_REQUIREMENTS, encoding="utf-8")
    (folder / "README.md").write_text(space_card(space_id, gradio_version), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--space", default="bhkr0901/defectspotter", help="<user>/<space-name>")
    parser.add_argument("--hardware", default="zero-a10g", help="zero-a10g (free ZeroGPU) or cpu-basic (needs PRO)")
    parser.add_argument("--private", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="only build the staging folder")
    args = parser.parse_args()

    import gradio
    from huggingface_hub import HfApi

    folder = Path(tempfile.mkdtemp(prefix="defectspotter_space_"))
    stage(folder, args.space, gradio.__version__)
    size_mb = sum(p.stat().st_size for p in folder.rglob("*") if p.is_file()) / 1024 ** 2
    print(f"Staged {size_mb:.1f} MB in {folder}")
    if args.dry_run:
        return

    api = HfApi(token=os.environ.get("HF_TOKEN") or None)
    user = api.whoami()["name"]
    owner = args.space.split("/")[0]
    print(f"Authenticated as {user}; deploying to {args.space}")
    if owner != user:
        print(f"Note: deploying to namespace '{owner}' as user '{user}' (must be an organisation you can write to).")
    try:
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", space_hardware=args.hardware,
                        private=args.private, exist_ok=True)
    except Exception as exc:  # hardware not allowed for this account -> create on default hardware
        print(f"Could not create with hardware={args.hardware}: {type(exc).__name__}: {exc}")
        print("Retrying without a hardware request (account default).")
        api.create_repo(args.space, repo_type="space", space_sdk="gradio", private=args.private, exist_ok=True)
    try:
        api.request_space_hardware(args.space, args.hardware)
    except Exception as exc:
        print(f"Hardware request '{args.hardware}' was not accepted: {exc}\n"
              "Free accounts can host up to 2 ZeroGPU Spaces (verified e-mail, account older than 30 days); "
              "cpu-basic Gradio Spaces need a PRO plan. Change hardware in the Space settings if needed.")
    api.upload_folder(repo_id=args.space, repo_type="space", folder_path=str(folder),
                      commit_message="Deploy DefectSpotter (app + ONNX models + evidence)")
    name = args.space.replace("/", "-").replace("_", "-").lower()
    print(f"\nSpace page : https://huggingface.co/spaces/{args.space}")
    print(f"Direct app : https://{name}.hf.space")
    print("The first build takes a few minutes (dependency install).")


if __name__ == "__main__":
    sys.exit(main())
