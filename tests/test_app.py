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


def test_detector_comparison_covers_all_five_detectors(inspector):
    from ui import explain as ex
    from ui.store import ReportStore

    store = ReportStore(ROOT)
    stats = ex.detector_stats(store, inspector)
    assert set(stats.method) == set(ex.DETECTOR_ORDER)
    assert stats.set_index("method").loc["patchcore", "rank"] == 1  # the deployed detector is the best one
    for column in ("auroc_mean", "pixel_auroc", "aupro", "cost_per_1000", "latency_ms"):
        assert stats[column].notna().all(), column
    html = ex.detector_comparison_html(store, inspector)
    assert "Model comparison" in html and "★" in html and "PatchCore is the best detector" in html
    assert "how they differ" in ex.detectors_html(store, inspector)


def test_build_id_is_stable_and_embedded():
    from ui.app import build_id
    from ui import explain as ex

    first, second = build_id(), build_id()
    assert first == second and first.startswith("ds-")
    assert f'data-build="{first}"' in ex.nav(first)


def test_guide_diagrams_are_valid_svg():
    import sys
    import xml.etree.ElementTree as ET

    sys.path.insert(0, str(ROOT / "scripts"))
    from guide_html import svg
    from guide_html.data import GuideData

    numbers = GuideData(ROOT).numbers()
    for name, make in svg.ALL.items():
        flow = make(numbers)
        ET.fromstring(flow.svg())  # well-formed XML
        for e in flow.edges:
            assert e["a"] in flow.nodes and e["b"] in flow.nodes, name


def test_detector_per_product_and_map(inspector):
    from ui import explain as ex
    from ui.store import ReportStore

    rows = ex.detector_per_product(ReportStore(ROOT), inspector)
    assert [r["product"].lower() for r in rows] == [c.lower() for c in inspector.categories]
    for r in rows:
        assert r["recommended"].startswith("PatchCore") and "PatchCore" in r["ranking"]
    html = ex.detector_map_html()
    assert "not trained here" in html and "no ResNet" in html


def test_deck_charts_are_valid_svg():
    import sys
    import xml.etree.ElementTree as ET

    sys.path.insert(0, str(ROOT / "scripts"))
    import build_deck

    ET.fromstring(build_deck.hbar([("a", 0.5, "#2a78d6", "a"), ("b", 1.0, "#6250d6", "b")], 1.0, lambda v: f"{v:.2f}", title="t"))
    ET.fromstring(build_deck.line_chart([(1, 0.7, 0.6, 0.8), (8, 0.97, 0.95, 0.99)], "t"))
