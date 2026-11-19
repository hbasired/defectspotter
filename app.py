"""DefectSpotter - entry point for local runs, Colab and Hugging Face Spaces.

Local:   python app.py            (then open http://127.0.0.1:7860)
Options: python app.py --share    (temporary public gradio.live link)
         python app.py --port 8080 --host 0.0.0.0
"""
from __future__ import annotations

# ZeroGPU Spaces need `spaces` imported before anything touches CUDA.
try:  # pragma: no cover - only present on Hugging Face Spaces
    import spaces  # type: ignore
except ImportError:  # local / Colab
    spaces = None

import argparse
import os
from pathlib import Path

ROOT = Path(os.environ.get("DEFECTSPOTTER_ROOT", Path(__file__).resolve().parent))


def ensure_artifacts(root: Path) -> None:
    """Download artifacts/reports/examples from a Hub model repo when they are not bundled."""
    if (root / "artifacts" / "manifest.json").exists():
        return
    repo_id = os.environ.get("MODEL_REPO_ID")
    if not repo_id:
        raise SystemExit("No trained artifacts found. Run `python scripts/train.py --data <mvtec>` "
                         "or set MODEL_REPO_ID to a Hugging Face repo that contains artifacts/.")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=repo_id, repo_type="model", local_dir=str(root),
                      allow_patterns=["artifacts/**", "reports/**", "examples/**"], token=os.environ.get("HF_TOKEN"))


if spaces is not None:  # ZeroGPU requires at least one @spaces.GPU function to be registered at start-up.
    @spaces.GPU(duration=20)
    def gpu_self_test() -> str:
        import torch

        return f"CUDA available: {torch.cuda.is_available()} · " + (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "no device")
else:
    gpu_self_test = None

ensure_artifacts(ROOT)

from ui.app import build_app, launch_kwargs  # noqa: E402  (after the optional `spaces` import)

demo = build_app(ROOT, gpu_check=gpu_self_test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the DefectSpotter app")
    parser.add_argument("--host", default=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("GRADIO_SERVER_PORT", 7860)))
    parser.add_argument("--share", action="store_true", help="create a temporary public gradio.live link")
    args = parser.parse_args()
    on_spaces = bool(os.environ.get("SPACE_ID"))
    demo.queue(default_concurrency_limit=4, max_size=64).launch(
        server_name="0.0.0.0" if on_spaces else args.host, server_port=args.port, share=args.share,
        show_error=True, allowed_paths=[str(ROOT / "examples"), str(ROOT / "reports")], **launch_kwargs())
