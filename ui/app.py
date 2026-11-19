"""DefectSpotter Gradio app: presentation, live inspection, batch, review queue, threshold/cost lab,
EDA, evaluation, drift monitor, few-shot onboarding and a JSON API."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import pandas as pd

from defectspotter.constants import CATEGORY_INFO, METHOD_LABELS, PROJECT_ROOT
from defectspotter.data import read_rgb, to_rgb_array
from defectspotter.decision import (CostModel, cost_curve, decide_many, operating_metrics, reference_costs,
                                    select_threshold_by_cost)
from defectspotter.drift import DRIFT_KINDS, apply_drift
from defectspotter.runtime import METHOD_ORDER, Inspector
from defectspotter.viz import fit_display, overlay

from . import charts
from .store import ReportStore
from .story import (cost_summary, data_html, decision_card, demo_guide_html, explanation_md, hero_html,
                    pipeline_html, problem_html, responsible_html, results_html)
from .theme import CSS, FORCE_LIGHT_JS, STATUS, STATUS_BG, STATUS_ICON, STATUS_TEXT, build_theme

GRADIO_MAJOR = int(gr.__version__.split(".")[0])
PRIVATE = {"api_visibility": "private"} if GRADIO_MAJOR >= 6 else {"api_name": False}
SHORT_METHOD = {"patchcore": "PatchCore", "vae": "VAE", "mean": "Mean", "ensemble": "Ensemble"}
QUEUE_COLUMNS = ["id", "time", "source", "file", "product", "detector", "score", "model_decision", "status"]
AUDIT_COLUMNS = ["time", "id", "file", "product", "detector", "score", "model_decision", "inspector_verdict", "note"]


def _labelled(choices: list[str], labels: dict) -> list[tuple[str, str]]:
    return [(labels.get(c, c), c) for c in choices]


def _thumb(image: np.ndarray, size: int = 320) -> np.ndarray:
    return fit_display(np.asarray(image), size)


def _health_html(flags: list[dict]) -> str:
    if not flags:
        return ""
    rows = "".join(
        f"<tr><td>{f['signal']}</td><td>{f['value']:.1f}</td><td>{f['reference_mean']:.1f}</td>"
        f"<td>{f['z']:+.1f}</td><td>{'✓ ok' if f['status'] == 'ok' else '⚠ out of range'}</td></tr>" for f in flags)
    bad = any(f["status"] != "ok" for f in flags)
    head = ("<div class='ds-warn'>⚠ Acquisition outside the qualified range — check lighting / focus / camera position."
            "</div>" if bad else "<div class='ds-ok'>✓ Camera health: image within the qualified acquisition range.</div>")
    return (f"{head}<table class='ds-table' style='margin-top:8px'><tr><th>signal</th><th>this image</th>"
            f"<th>qualified mean</th><th>z-score</th><th>status</th></tr>{rows}</table>")


def _stat_tiles(items: list[tuple[str, str, str | None]]) -> str:
    tiles = []
    for value, label, status in items:
        style = ""
        if status in STATUS:
            style = f"style='background:{STATUS_BG[status]};border-color:{STATUS[status]}'"
            value = f"{STATUS_ICON[status]} {value}"
        tiles.append(f"<div class='ds-kpi' {style}><div class='v'>{value}</div><div class='l'>{label}</div></div>")
    return f"<div class='ds-kpis' style='grid-template-columns:repeat({min(6, len(items))},minmax(0,1fr))'>{''.join(tiles)}</div>"


def build_app(project_root: str | Path = PROJECT_ROOT, gpu_check=None) -> gr.Blocks:
    """Build the full UI. ``gpu_check`` is an optional @spaces.GPU function (ZeroGPU Spaces only)."""
    root = Path(project_root)
    inspector = Inspector(root / "artifacts")
    store = ReportStore(root)
    categories = inspector.categories
    category_labels = {c: CATEGORY_INFO.get(c, {}).get("title", c) for c in categories}
    default_category = categories[0]
    default_method = "patchcore"
    default_policy = inspector.policy(default_category, default_method)
    metrics = store.csv("metrics.csv")

    # ------------------------------------------------------------ helpers
    def policy_text(category: str, method: str) -> str:
        cal = inspector.calibration(category)["methods"][method]
        pol = cal["policies"]
        lines = [f"**{category_labels[category]} · {METHOD_LABELS[method]}** — calibrated default: threshold "
                 f"**{pol['synthetic_cost']['threshold']:.1f}**, CHECK band **±{pol['synthetic_cost']['band']:.1f}** "
                 "(cost-optimal on held-out good parts + synthetic defects; no real defects used)."]
        if "real_oracle_cost" in pol:
            lines.append(f"Real-defect oracle would choose {pol['real_oracle_cost']['threshold']:.1f} ± "
                         f"{pol['real_oracle_cost']['band']:.1f}; 1%-false-reject quantile policy: "
                         f"{pol['good_quantile']['threshold']:.1f}.")
        return " ".join(lines)

    def on_category(category: str, method: str):
        methods = inspector.methods(category)
        method = method if method in methods else methods[0]
        pol = inspector.policy(category, method)
        return (gr.update(choices=_labelled(methods, METHOD_LABELS), value=method), pol["threshold"], pol["band"],
                policy_text(category, method), store.example_gallery(category), None)

    def on_method(category: str, method: str):
        pol = inspector.policy(category, method)
        return pol["threshold"], pol["band"], policy_text(category, method)

    def reset_policy(category: str, method: str):
        pol = inspector.policy(category, method)
        return pol["threshold"], pol["band"]

    # ------------------------------------------------------------ inspect
    def run_inspect(image, category, method, threshold, band, example):
        if image is None:
            raise gr.Error("Upload, paste or capture a part image — or click one of the samples.")
        result = inspector.inspect(image, category, method, float(threshold), float(band))
        truth, truth_img = None, None
        if example and example.get("category") == category:
            truth = "good part" if int(example["label"]) == 0 else f"defective — {example['defect_type'].replace('_', ' ')}"
            if example.get("mask"):
                mask = cv2.imread(str(store.examples / example["mask"]), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    display = result["display"]
                    mask = cv2.resize(mask, (display.shape[1], display.shape[0]), interpolation=cv2.INTER_NEAREST)
                    truth_img = display.copy()
                    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(truth_img, contours, -1, (42, 120, 214), max(2, display.shape[0] // 160))
        card = decision_card(result["decision"], result["score"], result["threshold"], result["band"], result["policy"],
                             result["timing"]["total_ms"], result["health_reasons"], truth)
        gallery = []
        for item in result["comparison"]:
            if "map" in item:
                gallery.append((overlay(result["display"], item["map"]),
                                f"{SHORT_METHOD[item['method']]} {item['score']:.0f} {STATUS_ICON.get(item['decision'], '')}{item['decision']}"))
        recon = None
        if result["reconstruction"] is not None:
            size = 256
            original = cv2.resize(to_rgb_array(image), (size, size), interpolation=cv2.INTER_AREA)
            rec = cv2.resize(result["reconstruction"], (size, size), interpolation=cv2.INTER_LINEAR)
            recon = np.concatenate([original, np.full((size, 6, 3), 255, np.uint8), rec], axis=1)
        record = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in result.items()
                  if k in ("category", "method", "decision", "score", "raw_score", "threshold", "band", "policy")}
        record["regions"] = result["regions"].to_dict(orient="records")
        record["timing_ms"] = {k: round(v, 1) for k, v in result["timing"].items()}
        record["health"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in f.items()} for f in result["health"]]
        record["detectors"] = [{"method": c["method"], "score": c["score"], "decision": c["decision"]}
                               for c in result["comparison"]]
        if truth:
            record["ground_truth"] = truth
        last = {"overlay": _thumb(result["overlay"]), "score": round(result["score"], 1), "decision": result["decision"],
                "category": category, "method": method,
                "file": (example or {}).get("file", "uploaded image") if example else "uploaded image"}
        return (card, charts.decision_bar(result["score"], result["threshold"], result["band"]), result["overlay"],
                truth_img if truth_img is not None else result["display"], recon, gallery,
                charts.anomaly_heatmap(result["anomaly_map"], image=result["display"]), result["regions"], _health_html(result["health"]),
                explanation_md(result), record, last)

    def live_inspect(frame, category, method, threshold, band):
        if frame is None:
            return None, ""
        r = inspector.inspect(frame, category, method, float(threshold), float(band), compare=False, display_size=384)
        return r["overlay"], decision_card(r["decision"], r["score"], r["threshold"], r["band"], "live camera",
                                           r["timing"]["total_ms"], r["health_reasons"])

    def select_example(category, evt: gr.SelectData):
        rows = store.examples_for(category)
        row = rows.iloc[int(evt.index)]
        example = {"category": category, "file": row.file, "label": int(row.label), "defect_type": row.defect_type,
                   "mask": row["mask"]}
        return str(store.examples / row.file), example

    # -------------------------------------------------------- review queue
    def queue_frame(queue: list[dict]) -> pd.DataFrame:
        if not queue:
            return pd.DataFrame(columns=QUEUE_COLUMNS)
        return pd.DataFrame([{k: item[k] for k in QUEUE_COLUMNS} for item in queue])

    def pending_choices(queue: list[dict]):
        ids = [f"#{q['id']} · {q['file']} · {q['score']:.0f}" for q in queue if q["status"] == "pending"]
        return gr.update(choices=ids, value=ids[0] if ids else None)

    def add_to_queue(queue: list[dict], items: list[dict], source: str) -> list[dict]:
        queue = list(queue or [])
        next_id = (max((q["id"] for q in queue), default=0) + 1)
        for item in items:
            queue.append({"id": next_id, "time": time.strftime("%H:%M:%S"), "source": source, "file": item["file"],
                          "product": item["category"], "detector": item["method"], "score": item["score"],
                          "model_decision": item["decision"], "status": "pending", "overlay": item["overlay"]})
            next_id += 1
        return queue

    def send_last_to_queue(queue, last):
        if not last:
            raise gr.Error("Inspect a part first.")
        queue = add_to_queue(queue, [last], "inspect")
        gr.Info(f"Added to the review queue ({sum(q['status'] == 'pending' for q in queue)} pending).")
        return queue, queue_frame(queue), pending_choices(queue), _queue_stats(queue)

    def _queue_stats(queue):
        pending = sum(q["status"] == "pending" for q in queue)
        confirmed = sum(q["status"] == "confirmed defect" for q in queue)
        released = sum(q["status"] == "released as good" for q in queue)
        return _stat_tiles([(str(pending), "pending CHECK decisions", "CHECK" if pending else None),
                            (str(confirmed), "confirmed defects (→ REJECT)", "REJECT" if confirmed else None),
                            (str(released), "released as good (→ PASS)", "PASS" if released else None),
                            (str(len(queue)), "items received", None)])

    def show_item(queue, choice):
        if not choice:
            return None
        item_id = int(choice.split("·")[0].strip().lstrip("#"))
        item = next((q for q in queue if q["id"] == item_id), None)
        return item["overlay"] if item else None

    def decide_item(queue, audit, choice, note, verdict):
        if not choice:
            raise gr.Error("Select a pending item.")
        item_id = int(choice.split("·")[0].strip().lstrip("#"))
        queue = [dict(q) for q in queue]
        audit = list(audit or [])
        for q in queue:
            if q["id"] == item_id:
                q["status"] = verdict
                audit.append({"time": time.strftime("%Y-%m-%d %H:%M:%S"), "id": item_id, "file": q["file"],
                              "product": q["product"], "detector": q["detector"], "score": q["score"],
                              "model_decision": q["model_decision"], "inspector_verdict": verdict, "note": note or ""})
        audit_df = pd.DataFrame(audit, columns=AUDIT_COLUMNS)
        path = Path(tempfile.mkdtemp(prefix="ds_audit_")) / "defectspotter_audit_log.csv"
        audit_df.to_csv(path, index=False)
        return (queue, audit, queue_frame(queue), pending_choices(queue), audit_df, str(path), "",
                _queue_stats(queue), None)

    # --------------------------------------------------------------- batch
    def load_sample_batch(category):
        return [str(store.examples / f) for f in store.examples_for(category).file]

    def run_batch(files, category, method, threshold, band, queue):
        if not files:
            raise gr.Error("Upload images or click 'Load sample batch' first.")
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
        names = [Path(p).name for p in paths]
        images = [read_rgb(p) for p in paths]
        frame, overlays = inspector.inspect_batch(images, names, category, method, float(threshold), float(band))
        counts = frame.decision.value_counts()
        tiles = _stat_tiles([(str(len(frame)), "parts inspected", None),
                             (str(counts.get("PASS", 0)), "PASS", "PASS"),
                             (str(counts.get("CHECK", 0)), "CHECK → review queue", "CHECK"),
                             (str(counts.get("REJECT", 0)), "REJECT", "REJECT"),
                             (f"{frame.latency_ms.median():.0f} ms", "median latency / part", None),
                             (f"{(frame.health != 'ok').sum()}", "camera-health warnings", None)])
        checks = [{"file": r.file, "category": category, "method": method, "score": r.score, "decision": r.decision,
                   "overlay": _thumb(ov[0])} for r, ov in zip(frame.itertuples(), overlays) if r.decision == "CHECK"]
        queue = add_to_queue(queue, checks, "batch") if checks else list(queue or [])
        path = Path(tempfile.mkdtemp(prefix="ds_batch_")) / f"defectspotter_batch_{category}.csv"
        frame.assign(product=category, detector=method).to_csv(path, index=False)
        return (tiles, charts.batch_strip(frame, float(threshold), float(band)), frame, overlays, str(path), queue,
                queue_frame(queue), pending_choices(queue), _queue_stats(queue))

    # --------------------------------------------------------- threshold lab
    def lab(category, method, threshold, band, prevalence, c_escape, c_fr, c_review, capacity, per_day, eval_set):
        costs = CostModel(defect_prevalence=prevalence / 100, cost_escape=c_escape, cost_false_reject=c_fr,
                          cost_review=c_review, review_capacity=capacity / 100, parts_per_day=int(per_day))
        g = store.score_groups(category, method)
        if not g:
            msg = charts.empty("No stored scores for this detector")
            return "", msg, msg, msg, pd.DataFrame(), pd.DataFrame()
        if eval_set.startswith("Operating"):
            mask = g["test_roles"] == "operating_test"
            scores, labels = g["test_scores"][mask], g["test_labels"][mask]
        else:
            scores = np.r_[g["cal_good"], g["synthetic"]]
            labels = np.r_[np.zeros(len(g["cal_good"])), np.ones(len(g["synthetic"]))]
        m = operating_metrics(scores, labels, float(threshold), float(band), costs)
        curve = cost_curve(scores, labels, float(band), costs)
        best = select_threshold_by_cost(scores, labels, costs, bands=(float(band),))
        refs = reference_costs(costs)
        per_year = m["expected_cost_per_1000"] / 1000 * costs.parts_per_day * 365
        manual_year = refs["manual_100pct_per_1000"] / 1000 * costs.parts_per_day * 365
        tiles = _stat_tiles([
            (f"{100 * m['false_reject_rate']:.1f}%", "false-reject rate (good → REJECT)", None),
            (f"{100 * m['escape_rate']:.1f}%", "escape rate (defect → PASS)", None),
            (f"{100 * m['review_rate']:.1f}%", "of production → CHECK (human)", None),
            (f"{m['expected_cost_per_1000']:,.0f}", "expected cost / 1,000 parts", None),
            (f"{per_year:,.0f}", f"per year at {costs.parts_per_day:,} parts/day", None),
            (f"{100 * (1 - per_year / manual_year):.0f}%" if manual_year else "—", "saving vs 100% manual inspection", None),
        ])
        groups = {"good parts (test)": g["test_scores"][(g["test_labels"] == 0)],
                  "defective parts (test)": g["test_scores"][(g["test_labels"] == 1)],
                  "synthetic defects (calibration)": g["synthetic"]}
        if "diffusion" in g:
            groups["diffusion defects (calibration)"] = g["diffusion"]
        dist = charts.score_distribution(groups, float(threshold), float(band), f"Score distributions · {category} · {METHOD_LABELS[method]}")
        decisions = decide_many(scores, float(threshold), float(band))
        matrix = pd.DataFrame([
            {"truth": truth, **{d: int(np.sum((labels == value) & (decisions == d))) for d in ("PASS", "CHECK", "REJECT")}}
            for truth, value in (("good part", 0), ("defective part", 1))])
        policies = inspector.calibration(category)["methods"][method]["policies"]
        rows = []
        test_mask = g["test_roles"] == "operating_test"
        for name, pol in {**policies, "your slider": {"threshold": float(threshold), "band": float(band)}}.items():
            mm = operating_metrics(g["test_scores"][test_mask], g["test_labels"][test_mask], pol["threshold"], pol["band"], costs)
            rows.append({"policy": name, "threshold": round(pol["threshold"], 1), "band": pol["band"],
                         "false_reject_%": round(100 * mm["false_reject_rate"], 1), "escape_%": round(100 * mm["escape_rate"], 1),
                         "review_%": round(100 * mm["review_rate"], 1), "cost_per_1000": round(mm["expected_cost_per_1000"], 1),
                         "uses_real_defects": "yes (oracle)" if "real" in name else "no"})
        return (tiles, dist, charts.rate_curves(curve, float(threshold)),
                charts.cost_curve(curve, float(threshold), best["threshold"], refs), matrix, pd.DataFrame(rows))

    def apply_optimal(category, method, band, prevalence, c_escape, c_fr, c_review, capacity):
        costs = CostModel(defect_prevalence=prevalence / 100, cost_escape=c_escape, cost_false_reject=c_fr,
                          cost_review=c_review, review_capacity=capacity / 100)
        g = store.score_groups(category, method)
        scores = np.r_[g["cal_good"], g["synthetic"]]
        labels = np.r_[np.zeros(len(g["cal_good"])), np.ones(len(g["synthetic"]))]
        best = select_threshold_by_cost(scores, labels, costs)
        gr.Info(f"Applied threshold {best['threshold']:.1f} ± {best['band']:.1f} (chosen with good + synthetic parts only).")
        return best["threshold"], best["band"]

    # ----------------------------------------------------------------- drift
    def drift_examples(category):
        rows = store.examples_for(category)
        return gr.update(choices=[(("good · " if r.label == 0 else f"{r.defect_type} · ") + Path(r.file).stem, r.file)
                                  for r in rows.itertuples()], value=rows.file.iloc[0] if len(rows) else None)

    def simulate_drift(category, method, file, kind, severity, threshold, band):
        if not file:
            raise gr.Error("Choose a sample part.")
        image = read_rgb(store.examples / file)
        base = inspector.inspect(image, category, method, float(threshold), float(band), compare=False)
        drifted_img = apply_drift(image, kind, float(severity) / 100)
        raw = inspector.inspect(drifted_img, category, method, float(threshold), float(band), compare=False,
                                escalate_on_health=False)
        gated = inspector.inspect(drifted_img, category, method, float(threshold), float(band), compare=False)
        sev = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        sweep = [inspector.inspect(apply_drift(image, kind, s), category, method, compare=False,
                                   escalate_on_health=False)["score"] for s in sev]
        card_base = decision_card(base["decision"], base["score"], base["threshold"], base["band"], "qualified camera",
                                  base["timing"]["total_ms"])
        card_raw = decision_card(raw["decision"], raw["score"], raw["threshold"], raw["band"],
                                 f"{kind.replace('_', ' ')} {severity:.0f}% · model only", raw["timing"]["total_ms"])
        card_gated = decision_card(gated["decision"], gated["score"], gated["threshold"], gated["band"],
                                   "model + camera-health gate", gated["timing"]["total_ms"], gated["health_reasons"])
        return (base["overlay"], raw["overlay"], card_base, card_raw, card_gated, _health_html(gated["health"]),
                charts.sweep_chart(sev, sweep, float(threshold), float(band), kind))

    def drift_views(category, threshold, band):
        stream = store.csv("drift_stream.csv")
        sens = store.csv("drift_sensitivity.csv")
        alarms = store.csv("drift_alarms.csv")
        alarms = alarms[alarms.category == category] if not alarms.empty else alarms
        reference = store.score_groups(category, "patchcore").get("cal_good", np.array([]))
        return (charts.drift_stream(stream, category, float(threshold), float(band)),
                charts.psi_chart(stream, category, reference),
                charts.health_stream(stream, category, "brightness"), charts.health_stream(stream, category, "sharpness"),
                alarms,
                charts.drift_heatmap(sens, category, "good_false_reject_rate", f"False-reject rate on GOOD parts under drift · {category}"),
                charts.drift_heatmap(sens, category, "health_flag_rate", f"Share caught by the camera-health gate · {category}"),
                charts.drift_heatmap(sens, category, "defect_escape_rate", f"Escape rate of real defects under drift · {category}"))

    # -------------------------------------------------------------- few-shot
    def onboard(files):
        if not files or len(files) < 2:
            raise gr.Error("Upload at least 2 good images of the new product (8+ recommended), or load the samples.")
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
        started = time.perf_counter()
        model = inspector.onboard_product([read_rgb(p) for p in paths], name="new product")
        seconds = time.perf_counter() - started
        summary = _stat_tiles([(str(model["n_good"]), "good images used", None),
                               (f"{model['bank_vectors']:,}", "memory-bank patches", None),
                               (f"{model['threshold']:.1f} ± {model['band']:.1f}", "threshold ± CHECK band", None),
                               (f"{seconds:.1f} s", "time to onboard (CPU)", None)])
        dist = charts.score_distribution({"good (leave-one-out)": model["good_scores"],
                                          "synthetic defects": model["synthetic_scores"]},
                                         model["threshold"], model["band"], "Calibration of the new product (no real defects)")
        return model, summary, dist, model["synthetic_examples"]

    def inspect_new(image, model):
        if model is None:
            raise gr.Error("Build the new-product model first (step 1).")
        if image is None:
            raise gr.Error("Upload or select a test image of the new product.")
        r = inspector.inspect_onboarded(image, model)
        return decision_card(r["decision"], r["score"], r["threshold"], r["band"], "few-shot onboarding", r["latency_ms"]), r["overlay"]

    # ----------------------------------------------------------------- API
    def api_inspect(image, category, method, threshold, band):
        if image is None:
            raise gr.Error("image is required")
        category = category if category in categories else default_category
        method = method if method in inspector.methods(category) else "patchcore"
        threshold = None if threshold in (None, "") or (isinstance(threshold, float) and np.isnan(threshold)) else float(threshold)
        band = None if band in (None, "") or (isinstance(band, float) and np.isnan(band)) else float(band)
        r = inspector.inspect(image, category, method, threshold, band)
        return ({"decision": r["decision"], "score": round(r["score"], 2), "threshold": r["threshold"], "band": r["band"],
                 "category": category, "method": method, "policy": r["policy"],
                 "regions": r["regions"].to_dict(orient="records"), "health": r["health"],
                 "detectors": [{"method": c["method"], "score": c["score"], "decision": c["decision"]} for c in r["comparison"]],
                 "latency_ms": round(r["timing"]["total_ms"], 1)}, r["overlay"])

    def api_batch(files, category, method):
        if not files:
            raise gr.Error("files are required")
        category = category if category in categories else default_category
        method = method if method in inspector.methods(category) else "patchcore"
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
        frame, _ = inspector.inspect_batch([read_rgb(p) for p in paths], [Path(p).name for p in paths], category, method)
        path = Path(tempfile.mkdtemp(prefix="ds_api_")) / "batch.csv"
        frame.to_csv(path, index=False)
        return frame.to_dict(orient="records"), str(path)

    def health():
        return {"status": "ok", "version": inspector.manifest.get("version"), "categories": categories,
                "methods": {c: inspector.methods(c) for c in categories},
                "feature_extractor": inspector.manifest.get("feature_extractor"),
                "policies": {c: {m: inspector.policy(c, m) for m in inspector.methods(c)} for c in categories}}

    # ================================================================ layout
    blocks_kwargs = {"title": "DefectSpotter · visual inspection without defect labels"}
    if GRADIO_MAJOR < 6:
        blocks_kwargs.update(theme=build_theme(), css=CSS, js=FORCE_LIGHT_JS)
    with gr.Blocks(**blocks_kwargs) as demo:
        queue_state = gr.State([])
        audit_state = gr.State([])
        example_state = gr.State(None)
        last_state = gr.State(None)
        onboard_state = gr.State(None)

        gr.HTML(hero_html(store, categories))
        with gr.Row(equal_height=True):
            category = gr.Dropdown(_labelled(categories, category_labels), value=default_category, label="Product type", scale=2)
            method = gr.Dropdown(_labelled(inspector.methods(default_category), METHOD_LABELS), value=default_method,
                                 label="Detector", scale=3)
            threshold = gr.Slider(0, 100, value=default_policy["threshold"], step=0.5, label="Decision threshold", scale=3,
                                  info="lower = fewer escapes, more false rejects")
            band = gr.Slider(0, 20, value=default_policy["band"], step=0.5, label="CHECK band ±", scale=3,
                             info="scores this close to the threshold go to a human")
            reset = gr.Button("↺ Calibrated policy", scale=1, size="sm")
        policy_md = gr.Markdown(policy_text(default_category, default_method), elem_classes="ds-note")

        with gr.Tabs():
            # ---------------------------------------------------------- story
            with gr.Tab("📘 Story"):
                gr.HTML(problem_html())
                gr.HTML(pipeline_html())
                with gr.Row():
                    gr.HTML(data_html(store))
                with gr.Row():
                    gr.Plot(charts.composition_chart(store.csv("dataset_composition.csv")), show_label=False)
                    gr.Plot(charts.metric_bars(metrics, "image_auroc", "Image AUROC"), show_label=False)
                gr.HTML(results_html(store))
                with gr.Row():
                    gr.Plot(charts.threshold_dumbbell(store.csv("threshold_comparison.csv"), "patchcore"), show_label=False)
                    cost_model = store.json("cost_model.json")
                    cs = cost_summary(store)
                    gr.Plot(charts.cost_comparison(cost_model, cs["ours"]) if cost_model and np.isfinite(cs["ours"])
                            else charts.empty("No cost model"), show_label=False)
                gr.HTML(responsible_html())
                gr.HTML(demo_guide_html())

            # -------------------------------------------------------- inspect
            with gr.Tab("🔍 Inspect"):
                with gr.Row():
                    with gr.Column(scale=5):
                        image = gr.Image(type="numpy", image_mode="RGB", sources=["upload", "webcam", "clipboard"],
                                         label="Part image · upload, paste or take a photo", height=360)
                        with gr.Row():
                            inspect_btn = gr.Button("🔍 Inspect part", variant="primary")
                            queue_btn = gr.Button("➕ Send to review queue")
                        examples = gr.Gallery(value=store.example_gallery(default_category), label="Samples from the "
                                              "held-out test set (click to inspect · ground truth is revealed after)",
                                              columns=5, height=270, allow_preview=False, object_fit="cover")
                    with gr.Column(scale=6):
                        card = gr.HTML(decision_card("PASS", 0, default_policy["threshold"], default_policy["band"],
                                                     "waiting for an image", 0).replace("✓ PASS", "Ready").replace(
                                                     "Within the learned envelope of good parts — ship.", "Inspect a part to get a decision."))
                        bar = gr.Plot(charts.decision_bar(0, default_policy["threshold"], default_policy["band"]), show_label=False)
                        with gr.Row():
                            overlay_img = gr.Image(label="Heat map + suspected regions", height=300, interactive=False)
                            truth_img = gr.Image(label="Input (blue outline = ground-truth defect for samples)", height=300,
                                                 interactive=False)
                with gr.Row():
                    compare = gr.Gallery(label="All detectors on this part · score · decision", columns=4, height=250,
                                         allow_preview=True, object_fit="contain")
                    recon = gr.Image(label="VAE: input (left) vs reconstruction of 'what a good part looks like' (right)",
                                     height=250, interactive=False)
                with gr.Row():
                    heat = gr.Plot(charts.anomaly_heatmap(None), show_label=False)
                    with gr.Column():
                        regions = gr.Dataframe(label="Suspected regions (pixel score > 50)", interactive=False)
                        health_box = gr.HTML()
                explanation = gr.Markdown()
                with gr.Accordion("📷 Live camera — continuous inspection at ~1 frame per second (demo, not a "
                                  "PLC-synchronised line loop)", open=False):
                    with gr.Row():
                        live_cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam", height=320)
                        live_overlay = gr.Image(label="Live heat map", height=320, interactive=False)
                        live_card = gr.HTML()
                with gr.Accordion("Machine-readable inspection record (same as the /inspect_part API)", open=False):
                    record = gr.JSON()

            # ---------------------------------------------------------- batch
            with gr.Tab("📦 Batch"):
                gr.Markdown("Inspect a whole tray or shift at once. CHECK decisions are routed automatically to the "
                            "**Review** tab; the full decision table can be downloaded as CSV for the MES/QMS.")
                with gr.Row():
                    batch_files = gr.File(file_count="multiple", file_types=["image"], label="Part images", scale=3)
                    with gr.Column(scale=1):
                        sample_btn = gr.Button("📂 Load sample batch for this product")
                        batch_btn = gr.Button("▶ Inspect batch", variant="primary")
                batch_tiles = gr.HTML()
                batch_chart = gr.Plot(charts.batch_strip(pd.DataFrame(), 50, 5), show_label=False)
                batch_table = gr.Dataframe(label="Batch summary", interactive=False)
                with gr.Row():
                    batch_gallery = gr.Gallery(label="Overlays", columns=6, height=320, scale=4)
                    batch_csv = gr.File(label="Download batch CSV", scale=1)

            # ---------------------------------------------------------- review
            with gr.Tab("🧑‍🔧 Review"):
                gr.Markdown("**Human in the loop.** Every CHECK decision waits here for a trained inspector. Your verdict, "
                            "note, the model score, threshold and detector are written to an audit log — the model is never "
                            "silently retrained from these decisions.")
                queue_tiles = gr.HTML(_queue_stats([]))
                queue_table = gr.Dataframe(queue_frame([]), label="Queue", interactive=False)
                with gr.Row():
                    with gr.Column(scale=2):
                        pending = gr.Dropdown([], label="Pending item")
                        note = gr.Textbox(label="Inspector note", placeholder="e.g. scratch 3 mm on rim, confirmed under loupe")
                        with gr.Row():
                            confirm_btn = gr.Button("✕ Confirm defect → REJECT", variant="stop")
                            release_btn = gr.Button("✓ Release as good → PASS")
                    review_img = gr.Image(label="Selected part", height=320, interactive=False, scale=2)
                audit_table = gr.Dataframe(pd.DataFrame(columns=AUDIT_COLUMNS), label="Audit log", interactive=False)
                audit_file = gr.File(label="Download audit log (CSV)")

            # --------------------------------------------------- threshold lab
            with gr.Tab("🎚️ Threshold & cost"):
                gr.Markdown("The threshold slider above trades **false rejects** against **escapes**. Here the trade-off is "
                            "priced: change the economics of *your* line and see the expected cost move. **Apply** recomputes "
                            "the cost-optimal threshold from good + synthetic parts only — exactly what a plant without "
                            "labelled defects can do.")
                with gr.Row():
                    prevalence = gr.Slider(0.1, 10, value=1.0, step=0.1, label="Defect rate (%)")
                    c_escape = gr.Number(400, label="Cost of an escape (per part)")
                    c_fr = gr.Number(2, label="Cost of a false reject (per part)")
                    c_review = gr.Number(0.30, label="Cost of a human CHECK (per part)")
                    capacity = gr.Slider(1, 30, value=10, step=1, label="Inspector capacity (% of parts)")
                    per_day = gr.Number(20000, label="Parts per day", precision=0)
                with gr.Row():
                    eval_set = gr.Radio(["Operating test split (real defects)", "Calibration (good + synthetic defects)"],
                                        value="Operating test split (real defects)", label="Evaluate on", scale=3)
                    apply_btn = gr.Button("⚙ Apply cost-optimal threshold (no real defects used)", variant="primary", scale=2)
                lab_tiles = gr.HTML()
                lab_dist = gr.Plot(show_label=False)
                with gr.Row():
                    lab_rates = gr.Plot(show_label=False)
                    lab_cost = gr.Plot(show_label=False)
                with gr.Row():
                    lab_matrix = gr.Dataframe(label="Decision matrix (truth × decision)", interactive=False)
                    lab_policies = gr.Dataframe(label="Threshold policies compared on the operating test split", interactive=False)
                gr.Markdown("**Label-free detection capability.** Before a synthetic set may influence the default threshold, the "
                            "detector must separate it from held-out good parts (synthetic AUROC ≥ 0.9). Sets below that are the "
                            "system's documented detection limit — like a measurement-capability study on a real line.")
                lab_capability = gr.Dataframe(label="Synthetic AUROC by calibration set (no real defects involved)",
                                              interactive=False)

            # ------------------------------------------------------------ EDA
            with gr.Tab("📊 EDA"):
                comp = store.csv("dataset_composition.csv")
                stats = store.csv("image_stats.csv")
                with gr.Row():
                    gr.Plot(charts.composition_chart(comp), show_label=False)
                    eda_types = gr.Plot(charts.defect_types_chart(comp, default_category), show_label=False)
                gr.Gallery([(p, f"{c}: good parts and every defect type (red = ground truth)") for c in
                            stats.category.unique() if (p := store.figure(f"eda_montage_{c}.jpg"))] if not stats.empty else [],
                           label="What the products and their defects look like", columns=2, height=430, object_fit="contain")
                with gr.Row():
                    signal = gr.Dropdown(["brightness", "contrast", "sharpness", "saturation_pct", "r_mean", "g_mean", "b_mean"],
                                         value="brightness", label="Acquisition signal", scale=1)
                with gr.Row():
                    eda_box = gr.Plot(charts.acquisition_box(stats, "brightness"), show_label=False)
                    gr.Plot(charts.mask_area_chart(stats), show_label=False)
                with gr.Row():
                    gr.Gallery([(p, f"where defects occur · {c}") for c in (stats.category.unique() if not stats.empty else [])
                                if (p := store.figure(f"eda_defect_location_{c}.png"))],
                               label="Defect location heat maps (sum of all ground-truth masks)", columns=4, height=260)
                    gr.Gallery([(p, f"mean | std · {c}") for c in categories if (p := store.figure(f"baseline_mean_std_{c}.jpg"))],
                               label="Mean good image and per-pixel variability (classical baseline)", columns=3, height=260)
                gr.Gallery(store.synthetic_gallery() + store.diffusion_gallery(),
                           label="Synthetic defects used to choose the threshold (cut-paste · scratch · stain · diffusion)",
                           columns=6, height=300)
                gr.Markdown(_eda_insights(stats, comp))
                gr.Dataframe(comp, label="Dataset composition", interactive=False)

            # ------------------------------------------------------ evaluation
            with gr.Tab("🧠 Models"):
                with gr.Row():
                    metric_choice = gr.Dropdown([("Image AUROC", "image_auroc"), ("Pixel AUROC", "pixel_auroc"),
                                                 ("AUPRO @ 30% FPR", "aupro_30"), ("Average precision", "image_ap")],
                                                value="image_auroc", label="Metric", scale=1)
                with gr.Row():
                    metric_plot = gr.Plot(charts.metric_bars(metrics, "image_auroc", "Image AUROC"), show_label=False)
                    roc_plot = gr.Plot(charts.roc_chart(store.json("roc_curves.json"), default_category), show_label=False)
                gr.Dataframe(metrics.round(4) if not metrics.empty else metrics,
                             label="All metrics (test split; CI = 95% stratified bootstrap)", interactive=False)
                with gr.Row():
                    defect_plot = gr.Plot(charts.defect_type_bars(store.csv("per_defect_type.csv"), default_category,
                                                                  default_method), show_label=False)
                    scatter_plot = gr.Plot(charts.method_scatter(store.json("scores.json"), default_category), show_label=False)
                qualitative = gr.Image(store.figure(f"qualitative_{default_category}.jpg"),
                                       label="Representative test parts (median-scoring example per defect type)",
                                       interactive=False, height=560)
                with gr.Row():
                    gr.Plot(charts.backbone_chart(store.csv("backbone_study.csv")), show_label=False)
                    gr.Plot(charts.latency_bars(store.csv("latency.csv")), show_label=False)
                with gr.Row():
                    gr.Plot(charts.accuracy_latency(metrics, store.csv("latency.csv")), show_label=False)
                    gr.Plot(charts.footprint_chart(store.csv("artifact_sizes.csv")), show_label=False)
                with gr.Row():
                    dumbbell = gr.Plot(charts.threshold_dumbbell(store.csv("threshold_comparison.csv"), default_method),
                                       show_label=False)
                    gr.Plot(charts.vae_history_chart(store.csv("vae_training_history.csv")), show_label=False)
                with gr.Accordion("Tables: backbone study · thresholds · latency · artifact sizes · hyper-parameter search", open=False):
                    gr.Dataframe(store.csv("backbone_study.csv").round(4), label="Backbone study (test split, for analysis only)")
                    gr.Dataframe(store.csv("threshold_comparison.csv").round(2), label="Synthetic vs real-defect thresholds")
                    gr.Dataframe(store.csv("latency.csv").round(1), label="Latency")
                    gr.Dataframe(store.csv("artifact_sizes.csv").round(2), label="Deployment artifacts (MB)")
                    gr.Dataframe(store.csv("hpo_patchcore.csv").round(4), label="PatchCore search (synthetic validation only)")
                    gr.Dataframe(store.csv("hpo_mean_baseline.csv").round(4), label="Baseline alignment search")

            # ----------------------------------------------------------- drift
            with gr.Tab("🌗 Drift"):
                gr.Markdown("**Lighting and camera changes look like defects to a good-only model.** Simulate a change on a "
                            "real part, then replay a production stream where the lamp ages and the camera is bumped.")
                with gr.Row():
                    drift_file = gr.Dropdown([], label="Part", scale=2)
                    drift_kind = gr.Radio([(v.split("(")[0].strip(), k) for k, v in DRIFT_KINDS.items()],
                                          value="lighting_dim", label="Acquisition change", scale=5)
                    severity = gr.Slider(0, 100, value=60, step=5, label="Severity (%)", scale=2)
                with gr.Row():
                    drift_base = gr.Image(label="Qualified camera", height=280, interactive=False)
                    drift_img = gr.Image(label="After the change", height=280, interactive=False)
                    with gr.Column():
                        drift_card_base = gr.HTML()
                        drift_card_raw = gr.HTML()
                        drift_card_gated = gr.HTML()
                with gr.Row():
                    drift_sweep = gr.Plot(show_label=False)
                    drift_health = gr.HTML()
                gr.Markdown("### Production stream replay (200 parts, ~3% defects)")
                stream_plot = gr.Plot(show_label=False)
                psi_plot = gr.Plot(show_label=False)
                with gr.Row():
                    bright_plot = gr.Plot(show_label=False)
                    sharp_plot = gr.Plot(show_label=False)
                alarm_table = gr.Dataframe(label="Part number of the first sustained alarm (2 in a row) after each change - "
                                                 "lamp ageing starts at part 61, maintenance at 121, camera bumped at 151",
                                           interactive=False)
                with gr.Row():
                    sens_frr = gr.Plot(show_label=False)
                    sens_flag = gr.Plot(show_label=False)
                    sens_escape = gr.Plot(show_label=False)
                gr.Markdown(MONITORING_POLICY)

            # -------------------------------------------------------- few-shot
            with gr.Tab("🆕 New product"):
                gr.Markdown("**Onboarding a new product line.** PatchCore needs no network training: a memory bank built from a "
                            "handful of good parts is enough. The curve shows the held-out metal-nut experiment; below you can "
                            "onboard it live.")
                with gr.Row():
                    gr.Plot(charts.few_shot_chart(store.csv("few_shot.csv")), show_label=False)
                    few = store.csv("few_shot.csv")
                    gr.Dataframe(few.groupby("shots", as_index=False)[["image_auroc", "pixel_auroc", "false_reject_rate",
                                                                        "escape_rate"]].mean().round(3) if not few.empty else few,
                                 label="Mean over random draws (FRR/escape at the leave-one-out threshold)", interactive=False)
                gr.Markdown("**Step 1 — upload good images of the new product** (or load 8 sample metal nuts) and build the model.")
                with gr.Row():
                    onboard_files = gr.File(file_count="multiple", file_types=["image"], label="Good images only", scale=3)
                    with gr.Column(scale=1):
                        onboard_samples = gr.Button("📂 Load 8 sample good metal nuts")
                        onboard_btn = gr.Button("🧱 Build model", variant="primary")
                onboard_summary = gr.HTML()
                with gr.Row():
                    onboard_dist = gr.Plot(show_label=False)
                    onboard_gallery = gr.Gallery(label="Synthetic defects generated for calibration", columns=3, height=300)
                gr.Markdown("**Step 2 — inspect new-product parts** (click a sample or upload).")
                with gr.Row():
                    new_samples = gr.Gallery([(p, Path(p).stem.split("_")[0]) for p in store.new_product_samples("test")],
                                             label="Held-out metal nut test parts", columns=5, height=220,
                                             allow_preview=False, scale=3)
                    new_image = gr.Image(type="numpy", label="New-product part", height=220, scale=2)
                with gr.Row():
                    new_card = gr.HTML()
                    new_overlay = gr.Image(label="Heat map", height=300, interactive=False)

            # ------------------------------------------------------------- API
            with gr.Tab("⚙️ API"):
                gr.Markdown(api_markdown(categories))
                with gr.Row():
                    health_json = gr.JSON(health(), label="/health")
                    gr.JSON(inspector.manifest, label="artifacts/manifest.json")
                health_btn = gr.Button("Refresh health")
                if gpu_check is not None:
                    gpu_btn = gr.Button("Run ZeroGPU self-test (inference itself runs on CPU via ONNX Runtime)")
                    gpu_out = gr.Textbox(label="GPU self-test")
                    gpu_btn.click(gpu_check, None, gpu_out, **PRIVATE)

        # Hidden components backing the public JSON API.
        with gr.Row(visible=False):
            api_image = gr.Image(type="numpy")
            api_category = gr.Textbox()
            api_method = gr.Textbox()
            api_threshold = gr.Number()
            api_band = gr.Number()
            api_out = gr.JSON()
            api_overlay = gr.Image()
            api_files = gr.File(file_count="multiple")
            api_batch_out = gr.JSON()
            api_batch_csv = gr.File()
            api_btn = gr.Button()
            api_batch_btn = gr.Button()

        # ============================================================ events
        inspect_outputs = [card, bar, overlay_img, truth_img, recon, compare, heat, regions, health_box, explanation,
                           record, last_state]
        inspect_inputs = [image, category, method, threshold, band, example_state]
        category.change(on_category, [category, method], [method, threshold, band, policy_md, examples, example_state], **PRIVATE)
        method.input(on_method, [category, method], [threshold, band, policy_md], **PRIVATE)
        reset.click(reset_policy, [category, method], [threshold, band], **PRIVATE)
        inspect_btn.click(run_inspect, inspect_inputs, inspect_outputs, **PRIVATE)
        image.input(lambda: None, None, example_state, **PRIVATE).then(run_inspect, inspect_inputs, inspect_outputs, **PRIVATE)
        examples.select(select_example, [category], [image, example_state], **PRIVATE).then(
            run_inspect, inspect_inputs, inspect_outputs, **PRIVATE)
        live_cam.stream(live_inspect, [live_cam, category, method, threshold, band], [live_overlay, live_card],
                        stream_every=1.0, time_limit=180, concurrency_limit=2, **PRIVATE)
        queue_btn.click(send_last_to_queue, [queue_state, last_state], [queue_state, queue_table, pending, queue_tiles], **PRIVATE)

        sample_btn.click(load_sample_batch, [category], [batch_files], **PRIVATE)
        batch_btn.click(run_batch, [batch_files, category, method, threshold, band, queue_state],
                        [batch_tiles, batch_chart, batch_table, batch_gallery, batch_csv, queue_state, queue_table, pending,
                         queue_tiles], **PRIVATE)

        pending.change(show_item, [queue_state, pending], [review_img], **PRIVATE)
        review_outputs = [queue_state, audit_state, queue_table, pending, audit_table, audit_file, note, queue_tiles, review_img]
        confirm_btn.click(lambda q, a, c, n: decide_item(q, a, c, n, "confirmed defect"), [queue_state, audit_state, pending, note],
                          review_outputs, **PRIVATE)
        release_btn.click(lambda q, a, c, n: decide_item(q, a, c, n, "released as good"), [queue_state, audit_state, pending, note],
                          review_outputs, **PRIVATE)

        lab_inputs = [category, method, threshold, band, prevalence, c_escape, c_fr, c_review, capacity, per_day, eval_set]
        lab_outputs = [lab_tiles, lab_dist, lab_rates, lab_cost, lab_matrix, lab_policies]
        gr.on([category.change, method.change, threshold.change, band.change, prevalence.change, c_escape.change,
               c_fr.change, c_review.change, capacity.change, per_day.change, eval_set.change, demo.load],
              lab, lab_inputs, lab_outputs, trigger_mode="always_last", **PRIVATE)
        def capability_table(category, method):
            frame = store.csv("detection_capability.csv")
            if frame.empty:
                return frame
            part = frame[(frame.category == category) & (frame.method == method)].copy()
            part["synthetic_auroc"] = part.synthetic_auroc.round(3)
            part["used"] = np.where(part.used, "✓ used for default threshold", "✕ below detection limit")
            return part[["calibration_set", "n", "synthetic_auroc", "used"]]

        gr.on([category.change, method.change, demo.load], capability_table, [category, method], [lab_capability], **PRIVATE)
        apply_btn.click(apply_optimal, [category, method, band, prevalence, c_escape, c_fr, c_review, capacity],
                        [threshold, band], **PRIVATE)

        signal.change(lambda s: charts.acquisition_box(store.csv("image_stats.csv"), s), [signal], [eda_box], **PRIVATE)
        metric_choice.change(lambda m: charts.metric_bars(metrics, m, {"image_auroc": "Image AUROC", "pixel_auroc": "Pixel AUROC",
                                                                       "aupro_30": "AUPRO @ 30% FPR", "image_ap": "Average precision"}[m]),
                             [metric_choice], [metric_plot], **PRIVATE)
        gr.on([category.change, method.change, demo.load],
              lambda c, m: (charts.defect_types_chart(store.csv("dataset_composition.csv"), c),
                            charts.roc_chart(store.json("roc_curves.json"), c),
                            charts.defect_type_bars(store.csv("per_defect_type.csv"), c, m),
                            charts.method_scatter(store.json("scores.json"), c),
                            store.figure(f"qualitative_{c}.jpg"),
                            charts.threshold_dumbbell(store.csv("threshold_comparison.csv"), m)),
              [category, method], [eda_types, roc_plot, defect_plot, scatter_plot, qualitative, dumbbell], **PRIVATE)

        gr.on([category.change, demo.load], drift_examples, [category], [drift_file], **PRIVATE)
        gr.on([category.change, threshold.change, band.change, demo.load], drift_views, [category, threshold, band],
              [stream_plot, psi_plot, bright_plot, sharp_plot, alarm_table, sens_frr, sens_flag, sens_escape],
              trigger_mode="always_last", **PRIVATE)
        gr.on([drift_file.change, drift_kind.change, severity.release, method.change],
              simulate_drift, [category, method, drift_file, drift_kind, severity, threshold, band],
              [drift_base, drift_img, drift_card_base, drift_card_raw, drift_card_gated, drift_health, drift_sweep],
              trigger_mode="always_last", **PRIVATE)

        onboard_samples.click(lambda: store.new_product_samples("good")[:8], None, [onboard_files], **PRIVATE)
        onboard_btn.click(onboard, [onboard_files], [onboard_state, onboard_summary, onboard_dist, onboard_gallery], **PRIVATE)
        new_samples.select(_select_path(store.new_product_samples("test")), None, [new_image], **PRIVATE).then(
            inspect_new, [new_image, onboard_state], [new_card, new_overlay], **PRIVATE)
        new_image.input(inspect_new, [new_image, onboard_state], [new_card, new_overlay], **PRIVATE)

        health_btn.click(health, None, [health_json], api_name="health")
        api_btn.click(api_inspect, [api_image, api_category, api_method, api_threshold, api_band], [api_out, api_overlay],
                      api_name="inspect_part")
        api_batch_btn.click(api_batch, [api_files, api_category, api_method], [api_batch_out, api_batch_csv],
                            api_name="inspect_batch")
    return demo


def _select_path(paths: list[str]):
    def handler(evt: gr.SelectData):
        return paths[int(evt.index)]
    return handler


def _eda_insights(stats: pd.DataFrame, comp: pd.DataFrame) -> str:
    if stats.empty:
        return ""
    lines = ["### What the EDA tells us"]
    defects = stats[(stats.label == 1) & (stats.mask_area_pct > 0)]
    if not defects.empty:
        small = (defects.mask_area_pct < 1).mean()
        lines.append(f"- **Tiny defects dominate:** {100 * small:.0f}% of labelled defects cover less than 1% of the image "
                     f"(median {defects.mask_area_pct.median():.2f}%). Down-scaling to 224 px is therefore a real risk, "
                     "which is why localisation metrics (pixel AUROC, AUPRO) are reported next to image AUROC.")
    by_cat = stats.groupby("category")
    for category, part in by_cat:
        lines.append(f"- **{category}:** {int(part.width.iloc[0])}×{int(part.height.iloc[0])} px, brightness "
                     f"{part.brightness.mean():.0f}±{part.brightness.std():.0f}, sharpness {part.sharpness.mean():.0f}±"
                     f"{part.sharpness.std():.0f}.")
    lines.append("- **Acquisition is tightly controlled** (small brightness spread within a product). That is what lets a "
                 "good-only model work — and why a lamp or focus change on the real line (Drift tab) is dangerous.")
    lines.append("- **Test sets are defect-heavy** (≈70% defective) unlike a real line (≈1%). We therefore report rates per class "
                 "and re-weight costs and review load to a 1% defect rate.")
    return "\n".join(lines)


MONITORING_POLICY = """
### Monitoring policy (what an operator would do)
| Signal | Alarm rule | Action |
|---|---|---|
| Camera health: brightness, contrast, focus | EWMA (λ = 0.2) outside ±3σ of the qualified images, 2 parts in a row | stop & check lamp/lens/camera mount, re-qualify with good parts |
| Anomaly-score stream | EWMA of scores above the upper control limit | treat as process **or** imaging drift: inspect recent CHECK/REJECTs |
| CHECK rate | > 2× the calibrated review load over a shift | inspector capacity alert; investigate drift before changing thresholds |
| Any single image out of qualified range | per-part health gate (|z| > 4 vs qualified images) | no automatic decision: PASS/REJECT → CHECK — never auto-ship or auto-scrap on an unqualified image |

Drift never triggers silent retraining: a memory-bank refresh is a controlled change with re-validation.
"""


def api_markdown(categories: list[str]) -> str:
    return f"""
### Use DefectSpotter from code
Three public endpoints (see **Use via API** in the footer for auto-generated docs):

| endpoint | inputs | returns |
|---|---|---|
| `/inspect_part` | image, category ({', '.join(categories)}), method (patchcore · vae · mean · ensemble), threshold, band (empty = calibrated) | JSON decision record + heat-map overlay |
| `/inspect_batch` | list of image files, category, method | per-part JSON rows + CSV |
| `/health` | – | versions, categories, calibrated policies |

```python
from gradio_client import Client, handle_file
client = Client("bhkr0901/defectspotter")          # or "http://127.0.0.1:7860" locally
record, overlay = client.predict(handle_file("part.png"), "bottle", "patchcore", None, None, api_name="/inspect_part")
print(record["decision"], record["score"])
```

### Model card (short)
- **Intended use:** research/teaching demo of good-only visual inspection with a human reviewing every CHECK.
- **Not for:** autonomous safety decisions, commercial use of MVTec AD data, unqualified or drifting cameras.
- **Detectors:** PatchCore-style memory bank on frozen ResNet-18 features (ONNX, ImageNet weights); convolutional VAE
  (ONNX); ECC-aligned mean-image baseline; PatchCore+VAE ensemble.
- **Calibration:** 0–100 score scale and thresholds from held-out good parts + synthetic defects only.
- **Data:** MVTec AD (CC BY-NC-SA 4.0), bottle / screw / carpet; metal nut for few-shot onboarding.
"""


def launch_kwargs() -> dict:
    if GRADIO_MAJOR >= 6:
        return {"theme": build_theme(), "css": CSS, "js": FORCE_LIGHT_JS}
    return {}
