"""Integration tests on the trained artifacts shipped in the repository (skipped if absent)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(not (ROOT / "artifacts" / "manifest.json").exists(),
                                reason="trained artifacts not present (run scripts/train.py)")


@pytest.fixture(scope="module")
def inspector():
    from defectspotter.runtime import Inspector

    return Inspector(ROOT / "artifacts")


def test_every_category_and_method_inspects(inspector):
    from ui.store import ReportStore

    store = ReportStore(ROOT)
    for category in inspector.categories:
        examples = store.examples_for(category)
        assert len(examples), f"no examples for {category}"
        image = ROOT / "examples" / examples.file.iloc[0]
        for method in inspector.methods(category):
            result = inspector.inspect(image, category, method)
            assert result["decision"] in {"PASS", "CHECK", "REJECT"}
            assert np.isfinite(result["score"])
            assert result["anomaly_map"].shape == (224, 224)
            assert result["overlay"].ndim == 3


def test_defective_sample_scores_higher_than_good(inspector):
    from ui.store import ReportStore

    store = ReportStore(ROOT)
    for category in inspector.categories:
        rows = store.examples_for(category)
        good = [inspector.inspect(ROOT / "examples" / f, category, compare=False)["score"] for f in rows[rows.label == 0].file]
        bad = [inspector.inspect(ROOT / "examples" / f, category, compare=False)["score"] for f in rows[rows.label == 1].file]
        assert np.median(bad) > np.median(good), category


def test_health_gate_routes_unqualified_images_to_check(inspector):
    from defectspotter.drift import apply_drift
    from defectspotter.data import read_rgb
    from ui.store import ReportStore

    category = inspector.categories[0]
    rows = ReportStore(ROOT).examples_for(category)
    image = apply_drift(read_rgb(ROOT / "examples" / rows.file.iloc[0]), "lighting_dim", 1.0)
    result = inspector.inspect(image, category)
    assert result["decision"] == "CHECK" and result["health_reasons"]


def test_few_shot_onboarding(inspector):
    from ui.store import ReportStore

    samples = ReportStore(ROOT).new_product_samples("good")[:4]
    if len(samples) < 2:
        pytest.skip("no new-product samples")
    model = inspector.onboard_product(samples, synthetic_per_image=1)
    result = inspector.inspect_onboarded(samples[0], model)
    assert result["decision"] in {"PASS", "CHECK", "REJECT"}


def test_app_builds():
    import gradio as gr

    from ui.app import build_app

    assert isinstance(build_app(ROOT), gr.Blocks)
