"""Unit tests for the numerical core (no trained artifacts needed)."""
from __future__ import annotations

import numpy as np
import pytest

from defectspotter.decision import (CostModel, decide, decide_many, fit_pixel_scale, fit_score_scale, normalize_map,
                                    normalize_scores, operating_metrics, reference_costs, select_threshold_by_cost,
                                    select_threshold_good_quantile)
from defectspotter.drift import DRIFT_KINDS, EWMAChart, apply_drift, first_alarm, health_check, population_stability_index
from defectspotter.metrics import aupro, bootstrap_auroc_ci, pixel_auroc, safe_auroc
from defectspotter.patchcore import MemoryBank, PatchCoreConfig, build_memory_bank, greedy_coreset, score_grid
from defectspotter.synthetic import SYNTHETIC_KINDS, make_synthetic_defect


# ------------------------------------------------------------------ PatchCore
def test_greedy_coreset_keeps_rare_cluster():
    """Greedy k-center must cover a rare cluster that random subsampling usually misses."""
    rng = np.random.default_rng(0)
    common = rng.normal(0, 0.1, size=(5000, 16))
    rare = rng.normal(5, 0.1, size=(5, 16))
    features = np.vstack([common, rare]).astype(np.float32)
    selected = greedy_coreset(features, 20, projection_dim=8, seed=1)
    assert len(set(selected.tolist())) == 20
    assert np.any(selected >= 5000), "the rare cluster must be represented"


def test_memory_bank_matches_brute_force():
    rng = np.random.default_rng(1)
    bank = rng.normal(size=(300, 12)).astype(np.float32)
    queries = rng.normal(size=(50, 12)).astype(np.float32)
    distances, index = MemoryBank(bank, chunk_size=64).nearest(queries)
    brute = np.sqrt(((queries[:, None, :] - bank[None]) ** 2).sum(-1))
    np.testing.assert_allclose(distances, brute.min(1), rtol=1e-4, atol=1e-4)
    np.testing.assert_array_equal(index, brute.argmin(1))


def test_score_grid_flags_the_odd_patch():
    rng = np.random.default_rng(2)
    good = rng.normal(0, 0.05, size=(4, 8, 8, 6)).astype(np.float32)
    bank = MemoryBank(build_memory_bank(good, coreset_ratio=1.0))
    test = rng.normal(0, 0.05, size=(8, 8, 6)).astype(np.float32)
    test[2, 5] += 3.0
    score, amap, patches = score_grid(test, bank, PatchCoreConfig(image_size=32, blur_sigma=1.0))
    assert np.unravel_index(patches.argmax(), patches.shape) == (2, 5)
    assert amap.shape == (32, 32) and score == pytest.approx(patches.max())


# ------------------------------------------------------------------ decisions
def test_three_way_rule():
    assert decide(10, 50, 5) == "PASS"
    assert decide(45, 50, 5) == "CHECK"
    assert decide(54.9, 50, 5) == "CHECK"
    assert decide(55, 50, 5) == "REJECT"
    assert list(decide_many([10, 50, 90], 50, 5)) == ["PASS", "CHECK", "REJECT"]


def test_operating_metrics_cost_identity():
    scores = np.array([10, 20, 30, 48, 70, 90.0])
    labels = np.array([0, 0, 0, 1, 1, 1])
    costs = CostModel()
    m = operating_metrics(scores, labels, 50, 5, costs)
    assert m["false_reject_rate"] == 0 and m["escape_rate"] == 0
    assert m["expected_cost_per_1000"] == pytest.approx(m["false_reject_cost_per_1000"] + m["escape_cost_per_1000"]
                                                        + m["review_cost_per_1000"])


def test_cost_threshold_respects_review_capacity():
    rng = np.random.default_rng(3)
    scores = np.r_[rng.normal(25, 5, 200), rng.normal(70, 10, 200)]
    labels = np.r_[np.zeros(200), np.ones(200)]
    costs = CostModel(review_capacity=0.05)
    best = select_threshold_by_cost(scores, labels, costs)
    assert best["review_rate"] <= 0.05 + 1e-9
    assert 25 < best["threshold"] < 70


def test_score_scale_anchors():
    scale = fit_score_scale([1.0, 1.1, 0.9, 1.0], [3.0, 3.2, 2.8])
    assert normalize_scores(1.0, scale) == pytest.approx(25)
    assert normalize_scores(3.0, scale) == pytest.approx(75)


def test_pixel_scale_piecewise():
    rng = np.random.default_rng(4)
    good = [rng.normal(1, 0.1, (16, 16)) for _ in range(5)]
    scale = fit_pixel_scale(good)
    assert normalize_map(np.array(scale["low"]), scale) == pytest.approx(0, abs=1e-3)
    assert normalize_map(np.array(scale["threshold"]), scale) == pytest.approx(50, abs=1e-3)
    assert normalize_map(np.array(scale["high"]), scale) == pytest.approx(100, abs=1e-3)


def test_good_quantile_and_reference_costs():
    t = select_threshold_good_quantile(np.linspace(0, 100, 101), 0.01, 5)
    assert t == pytest.approx(94.0)
    refs = reference_costs(CostModel())
    assert refs["no_inspection_per_1000"] > refs["manual_100pct_per_1000"] > 0


# -------------------------------------------------------------------- metrics
def test_metrics_perfect_and_random():
    labels = np.r_[np.zeros(50), np.ones(50)]
    assert safe_auroc(labels, labels) == 1.0
    low, high = bootstrap_auroc_ci(labels, labels + 0.01, n_boot=50)
    assert low <= high <= 1.0
    masks = [np.zeros((32, 32), np.uint8) for _ in range(3)]
    masks[1][8:16, 8:16] = 1
    maps = [m.astype(np.float32) for m in masks]
    assert pixel_auroc(maps, masks) == 1.0
    assert aupro(maps, masks) == pytest.approx(1.0, abs=0.02)


# ------------------------------------------------------------------ synthetic
@pytest.mark.parametrize("kind", SYNTHETIC_KINDS)
def test_synthetic_defects(kind):
    rng = np.random.default_rng(5)
    image = (rng.normal(128, 10, (96, 96, 3))).clip(0, 255).astype(np.uint8)
    out, mask, chosen = make_synthetic_defect(image, "carpet", seed=11, kind=kind)
    assert out.shape == image.shape and mask.shape == image.shape[:2]
    assert mask.sum() > 0 and chosen == kind
    assert np.abs(out.astype(int) - image.astype(int))[mask > 0].mean() > 1


# ---------------------------------------------------------------------- drift
@pytest.mark.parametrize("kind", list(DRIFT_KINDS))
def test_drift_perturbations(kind):
    image = np.full((64, 64, 3), 120, np.uint8)
    image[16:48, 16:48] = 200
    assert np.array_equal(apply_drift(image, kind, 0.0), image)
    changed = apply_drift(image, kind, 1.0)
    assert changed.shape == image.shape and not np.array_equal(changed, image)


def test_ewma_alarm_and_health_check():
    values = np.r_[np.random.default_rng(6).normal(0, 1, 50), np.full(30, 3.0)]
    chart = EWMAChart(0.0, 1.0).run(values)
    alarm = first_alarm(chart, 50)
    assert alarm is not None and 50 < alarm <= 60
    flags = health_check({"brightness": 40.0}, {"brightness": {"mean": 120.0, "std": 5.0}})
    assert flags[0]["status"] == "OUT OF RANGE"
    assert population_stability_index(np.arange(100), np.arange(100) + 50) > 0.25


# ------------------------------------------------------------------ pose normalisation
def test_pose_alignment_keeps_part_in_canvas_and_maps_back_exactly():
    """A long synthetic 'screw' (bar + wide head) is rotated head-left without leaving the canvas, and a map point
    warped to the aligned frame and back lands where it started."""
    import cv2
    from defectspotter.pose import align_image, part_pose, unalign_map

    img = np.full((512, 512, 3), 200, np.uint8)
    cv2.line(img, (40, 330), (330, 120), (40, 40, 40), 18)          # shank: diagonal, off-centre (~360 px long)
    cv2.circle(img, (330, 120), 30, (40, 40, 40), -1)               # wide head at the upper-right end
    pose = part_pose(img)
    assert pose["elongation"] > 2.0
    aligned, matrix = align_image(img)
    dark = (aligned.mean(axis=2) < 120)
    border = np.ones_like(dark)
    border[3:-3, 3:-3] = False
    assert not (dark & border).any()                                 # nothing pushed off / onto the canvas edge
    assert dark.sum() >= 0.9 * (img.mean(axis=2) < 120).sum()       # (bilinear rotation softens the outline)
    cols = np.nonzero(dark.any(axis=0))[0]
    left, right = dark[:, cols[0]:cols[0] + 40].sum(), dark[:, cols[-1] - 40:cols[-1]].sum()
    assert left > right                                              # head (wider end) on the left
    size = 128
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    p = np.array([200.0, 300.0, 1.0])
    q = matrix @ p
    centre = ((q[0] + 0.5) * size / 512 - 0.5, (q[1] + 0.5) * size / 512 - 0.5)
    blob = np.exp(-((xx - centre[0]) ** 2 + (yy - centre[1]) ** 2) / 4.0).astype(np.float32)
    back = unalign_map(blob, matrix, img.shape, size)
    w = back / back.sum()
    expected = ((p[1] + 0.5) * size / 512 - 0.5, (p[0] + 0.5) * size / 512 - 0.5)
    assert abs((w * yy).sum() - expected[0]) < 0.05 and abs((w * xx).sum() - expected[1]) < 0.05


# ------------------------------------------------------------------ real-world shift view
def test_shift_view_turns_rates_into_counts_people_and_money():
    from defectspotter.operations import LineScenario, manual_inspection_view, shift_view

    sc = LineScenario(parts_per_hour=1000, shift_hours=8, defect_rate=0.01, seconds_per_check=36,
                      inspector_cost_per_hour=30, human_detection_rate=0.8, cost_false_reject=2, cost_escape=400)
    assert sc.parts_per_shift == 8000 and sc.cost_per_check == pytest.approx(0.30)
    v = shift_view({"false_reject_rate": 0.0, "good_check_rate": 0.1, "escape_rate": 0.05,
                    "defect_check_rate": 0.2, "defect_reject_rate": 0.75}, sc)
    assert v["to_human"] == pytest.approx(7920 * 0.1 + 80 * 0.2)               # 808 parts to a human
    assert v["inspector_hours"] == pytest.approx(808 * 36 / 3600)
    assert v["defects_escaping"] == pytest.approx(80 * 0.05 + 80 * 0.2 * 0.2)   # passed + missed by the human
    assert v["cost_per_shift"] == pytest.approx(7.2 * 400 + 808 * 0.30)
    m = manual_inspection_view(sc)
    assert m["inspectors_needed"] == pytest.approx(10) and m["defects_escaping"] == pytest.approx(16)
