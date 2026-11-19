"""Project-wide constants and product metadata."""
from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
DEFAULT_REPORT_DIR = PROJECT_ROOT / "reports"
DEFAULT_EXAMPLE_DIR = PROJECT_ROOT / "examples"

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# Product lines used for the main experiment. ``metal_nut`` is held back as a
# "new product" for the few-shot onboarding experiment.
MAIN_CATEGORIES = ("bottle", "screw", "carpet")
FEW_SHOT_CATEGORY = "metal_nut"
TEXTURE_CATEGORIES = {"carpet", "grid", "leather", "tile", "wood"}

# Mirror of MVTec AD on the Hugging Face Hub, pinned to the commit that
# completed the upload so every run downloads exactly the same files.
MVTEC_HF_REPO = "foersben/mvtec-ad"
MVTEC_HF_REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"
MVTEC_OFFICIAL_URL = "https://www.mvtec.com/research-teaching/datasets/mvtec-ad"
MVTEC_LICENSE = "CC BY-NC-SA 4.0 (non-commercial research and teaching only)"
MVTEC_CITATION = (
    "Bergmann, Batzner, Fauser, Sattlegger, Steger. The MVTec Anomaly Detection Dataset: "
    "A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection. IJCV 129(4), 2021."
)

DECISIONS = ("PASS", "CHECK", "REJECT")
DECISION_COLORS = {"PASS": "#16a34a", "CHECK": "#d97706", "REJECT": "#dc2626"}

METHOD_LABELS = {
    "patchcore": "PatchCore (CNN memory bank)",
    "vae": "Convolutional VAE",
    "mean": "Aligned mean-image baseline",
    "ensemble": "PatchCore + VAE ensemble",
}

CATEGORY_INFO = {
    "bottle": {
        "title": "Bottle",
        "kind": "object",
        "summary": "Glass bottle seen from above: broken rims and contamination inside the neck.",
        "line_context": "Filling line, top-down camera before capping.",
    },
    "screw": {
        "title": "Screw",
        "kind": "object",
        "summary": "Small metal screw in random orientation: damaged threads, scratched head, chipped tip.",
        "line_context": "Fastener sorting station, back-lit tray camera.",
    },
    "carpet": {
        "title": "Carpet",
        "kind": "texture",
        "summary": "Woven carpet texture: holes, cuts, colour spots, metal contamination and loose threads.",
        "line_context": "Web inspection of continuous textile coming off the loom.",
    },
    "metal_nut": {
        "title": "Metal nut",
        "kind": "object",
        "summary": "Machined nut: bent flaps, colour defects, scratches and parts flipped upside-down.",
        "line_context": "New product onboarded from only a handful of good parts.",
    },
}
