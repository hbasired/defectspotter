"""DefectSpotter Gradio app — a presentation that ends in a live demonstration.

Tabs (left to right): Overview · Data · Models & training · Results · Threshold & cost · Drift & new lots ·
New product · Guide · 🎬 Live demo (inspect · batch · review queue · API). Every page opens with a short
"what you see / how to read it" box and every chart has a caption, so the app explains itself.
"""
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
from defectspotter.decision import (CostModel, calibration_sets, cost_curve, decide_many, default_policy,
                                    operating_metrics, reference_costs, select_threshold_by_cost)
from defectspotter.drift import DRIFT_KINDS, apply_drift
from defectspotter.runtime import Inspector
from defectspotter.viz import fit_display, overlay

from . import charts
from . import explain as ex
from .store import ReportStore
from .story import (cost_summary, decision_card, explanation_md, pipeline_html, problem_html, responsible_html,
                    results_html)
from .theme import CSS, FORCE_LIGHT_JS, STATUS, STATUS_BG, STATUS_ICON, build_theme

GRADIO_MAJOR = int(gr.__version__.split(".")[0])
PRIVATE = {"api_visibility": "private"} if GRADIO_MAJOR >= 6 else {"api_name": False}
SHORT_METHOD = {"patchcore": "PatchCore", "fvae": "Feature VAE", "vae": "Pixel VAE", "mean": "Mean", "ensemble": "Ensemble"}
QUEUE_COLUMNS = ["id", "time", "source", "file", "product", "detector", "score", "model_decision", "status"]
AUDIT_COLUMNS = ["time", "id", "file", "product", "detector", "score", "model_decision", "inspector_verdict", "note"]
POLICY_NAMES = {"synthetic_cost": "calibrated without real defects", "real_oracle_cost": "oracle (uses real defects)",
                "good_quantile": "1% false-reject quantile", "re-qualified (current lot)": "re-qualified for the current lot"}
LOT_TRAIN = "Training lot (qualified)"
LOT_CURRENT = "Current lot (re-qualified)"


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
            f"<th>qualified mean</th><th>z-score</th><th>status</th></tr>{rows}</table>"
            "<div class='ds-small'>z-score = how many standard deviations this image is from the images the model was "
            "qualified on; beyond ±4 no automatic decision is taken.</div>")


def _stat_tiles(items: list[tuple[str, str, str | None]]) -> str:
    tiles = []
    for value, label, status in items:
        style = ""
        if status in STATUS:
            style = f"style='background:{STATUS_BG[status]};border-color:{STATUS[status]}'"
            value = f"{STATUS_ICON[status]} {value}"
        tiles.append(f"<div class='ds-kpi' {style}><div class='v'>{value}</div><div class='l'>{label}</div></div>")
    return f"<div class='ds-kpis' style='grid-template-columns:repeat({min(6, len(items))},minmax(0,1fr))'>{''.join(tiles)}</div>"


def _cap(text: str):
    return gr.HTML(ex.caption(text))


def _section(text: str):
    return gr.HTML(f"<div class='ds-section'>{text}</div>")


def build_app(project_root: str | Path = PROJECT_ROOT, gpu_check=None) -> gr.Blocks:
    """Build the full UI. ``gpu_check`` is an optional @spaces.GPU function (ZeroGPU Spaces only)."""
    root = Path(project_root)
    inspector = Inspector(root / "artifacts")
    store = ReportStore(root)
    categories = inspector.categories
    category_labels = {c: CATEGORY_INFO.get(c, {}).get("title", c) for c in categories}
    default_category = categories[0]
    default_method = "patchcore"
    metrics = store.csv("metrics.csv")
    guide_path = root / "docs" / "GUIDE.md"

    # ------------------------------------------------------------ lot helpers
    def lot_available(category: str, session_lots: dict | None) -> bool:
        return bool((session_lots or {}).get(category)) or inspector.shipped_lot(category) is not None

    def active_lot(category: str, method: str, lot_choice: str | None, session_lots: dict | None):
        if method != "patchcore" or lot_choice != LOT_CURRENT:
            return None
        return (session_lots or {}).get(category) or inspector.shipped_lot(category)

    def lot_radio(category: str, session_lots: dict | None, prefer: str | None = None):
        if lot_available(category, session_lots):
            return gr.update(choices=[LOT_TRAIN, LOT_CURRENT], value=prefer or LOT_CURRENT, interactive=True)
        return gr.update(choices=[LOT_TRAIN], value=LOT_TRAIN, interactive=False)

    def current_policy(category, method, lot_choice, session_lots) -> dict:
        lot = active_lot(category, method, lot_choice, session_lots)
        return inspector.lot_policy(lot) if lot else inspector.policy(category, method)

    def policy_text(category: str, method: str, lot_choice: str | None = None, session_lots: dict | None = None) -> str:
        lot = active_lot(category, method, lot_choice, session_lots)
        if lot:
            shift = lot.get("lot_shift", {})
            return (f"**{category_labels[category]} · {METHOD_LABELS[method]} · current lot** — re-qualified with "
                    f"**{lot.get('n_good', '?')} good parts of the current lot**: threshold **{lot['threshold']:.1f}**, "
                    f"CHECK band **±{lot['band']:.1f}** (lot-shift test p = {shift.get('p_value', float('nan')):.3g}; "
                    "chosen with the new good parts + synthetic defects made from them — no real defects).")
        pol = inspector.calibration(category)["methods"][method]["policies"]
        text = (f"**{category_labels[category]} · {METHOD_LABELS[method]}** — calibrated default: threshold "
                f"**{pol['synthetic_cost']['threshold']:.1f}**, CHECK band **±{pol['synthetic_cost']['band']:.1f}** "
                "(cost-optimal on held-out good parts + synthetic defects; no real defects used).")
        if "real_oracle_cost" in pol:
            text += (f" For comparison, a threshold chosen *with* real defect labels would be "
                     f"{pol['real_oracle_cost']['threshold']:.1f} ± {pol['real_oracle_cost']['band']:.1f}.")
        return text

    def on_category(category: str, method: str, session_lots):
        methods = inspector.methods(category)
        method = method if method in methods else methods[0]
        lot_choice = LOT_CURRENT if lot_available(category, session_lots) else LOT_TRAIN
        pol = current_policy(category, method, lot_choice, session_lots)
        return (gr.update(choices=_labelled(methods, METHOD_LABELS), value=method), lot_radio(category, session_lots),
                pol["threshold"], pol["band"], policy_text(category, method, lot_choice, session_lots),
                store.example_gallery(category), None)

    def on_method_or_lot(category: str, method: str, lot_choice: str, session_lots):
        pol = current_policy(category, method, lot_choice, session_lots)
        return pol["threshold"], pol["band"], policy_text(category, method, lot_choice, session_lots)

    def reset_policy(category: str, method: str, lot_choice: str, session_lots):
        pol = current_policy(category, method, lot_choice, session_lots)
        return pol["threshold"], pol["band"]

    # ------------------------------------------------------------ inspect
    def run_inspect(image, category, method, threshold, band, example, lot_choice, session_lots):
        if image is None:
            raise gr.Error("Upload, paste or capture a part image — or click one of the samples.")
        lot = active_lot(category, method, lot_choice, session_lots)
        result = inspector.inspect(image, category, method, float(threshold), float(band), lot=lot)
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
        card = decision_card(result["decision"], result["score"], result["threshold"], result["band"],
                             POLICY_NAMES.get(result["policy"], result["policy"]), result["timing"]["total_ms"],
                             result["health_reasons"], truth)
        gallery = []
        for item in result["comparison"]:
            if "map" in item:
                gallery.append((overlay(result["display"], item["map"]),
                                f"{SHORT_METHOD.get(item['method'], item['method'])} {item['score']:.0f} "
                                f"{STATUS_ICON.get(item['decision'], '')}{item['decision']}"))
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
                charts.anomaly_heatmap(result["anomaly_map"], image=result["display"]), result["regions"],
                _health_html(result["health"]), explanation_md(result), record, last)

    def live_inspect(frame, category, method, threshold, band, lot_choice, session_lots):
        if frame is None:
            return None, ""
        lot = active_lot(category, method, lot_choice, session_lots)
        r = inspector.inspect(frame, category, method, float(threshold), float(band), compare=False, display_size=384, lot=lot)
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

    def _queue_stats(queue):
        pending = sum(q["status"] == "pending" for q in queue)
        confirmed = sum(q["status"] == "confirmed defect" for q in queue)
        released = sum(q["status"] == "released as good" for q in queue)
        return _stat_tiles([(str(pending), "pending CHECK decisions", "CHECK" if pending else None),
                            (str(confirmed), "confirmed defects (→ REJECT)", "REJECT" if confirmed else None),
                            (str(released), "released as good (→ PASS)", "PASS" if released else None),
                            (str(len(queue)), "items received", None)])

    def send_last_to_queue(queue, last):
        if not last:
            raise gr.Error("Inspect a part first.")
        queue = add_to_queue(queue, [last], "inspect")
        gr.Info(f"Added to the review queue ({sum(q['status'] == 'pending' for q in queue)} pending).")
        return queue, queue_frame(queue), pending_choices(queue), _queue_stats(queue)

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

    def run_batch(files, category, method, threshold, band, queue, lot_choice, session_lots):
        if not files:
            raise gr.Error("Upload images or click 'Load sample batch' first.")
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
        names = [Path(p).name for p in paths]
        images = [read_rgb(p) for p in paths]
        lot = active_lot(category, method, lot_choice, session_lots)
        frame, overlays = inspector.inspect_batch(images, names, category, method, float(threshold), float(band), lot=lot)
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
            return "", msg, msg, msg, msg, pd.DataFrame()
        if eval_set.startswith("Operating"):
            mask = g["test_roles"] == "operating_test"
            scores, labels = g["test_scores"][mask], g["test_labels"][mask]
        else:  # label-free calibration data exactly as used for the default policy
            defects, _ = calibration_sets(store.calibration_groups(g))
            scores = np.r_[g["cal_good"], defects]
            labels = np.r_[np.zeros(len(g["cal_good"])), np.ones(len(defects))]
        m = operating_metrics(scores, labels, float(threshold), float(band), costs)
        curve = cost_curve(scores, labels, float(band), costs)
        best = select_threshold_by_cost(scores, labels, costs, bands=(float(band),),
                                        smooth=not eval_set.startswith("Operating"))
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
        matrix = charts.confusion_heatmap(charts.confusion_counts(scores, labels, float(threshold), float(band)),
                                          f"Confusion matrix at your threshold · {eval_set.split(' (')[0].lower()}")
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
        # Same label-free recipe as the deployed default (decision.default_policy).
        best = default_policy(store.calibration_groups(g), costs)
        gr.Info(f"Applied threshold {best['threshold']:.1f} ± {best['band']:.1f} (chosen with good + synthetic parts only).")
        return best["threshold"], best["band"]

    def capability_table(category, method):
        frame = store.csv("detection_capability.csv")
        if frame.empty:
            return frame
        part = frame[(frame.category == category) & (frame.method == method)].copy()
        part["synthetic_auroc"] = part.synthetic_auroc.round(3)
        part["used"] = np.where(part.used, "✓ used for default threshold", "✕ below detection limit")
        return part[["calibration_set", "n", "synthetic_auroc", "used"]]

    # ----------------------------------------------------------------- drift
    def drift_examples(category):
        rows = store.examples_for(category)
        return gr.update(choices=[(("good · " if r.label == 0 else f"{r.defect_type} · ") + Path(r.file).stem, r.file)
                                  for r in rows.itertuples()], value=rows.file.iloc[0] if len(rows) else None)

    def simulate_drift(category, method, file, kind, severity, threshold, band, lot_choice, session_lots):
        if not file:
            raise gr.Error("Choose a sample part.")
        lot = active_lot(category, method, lot_choice, session_lots)
        image = read_rgb(store.examples / file)
        base = inspector.inspect(image, category, method, float(threshold), float(band), compare=False, lot=lot)
        drifted_img = apply_drift(image, kind, float(severity) / 100)
        raw = inspector.inspect(drifted_img, category, method, float(threshold), float(band), compare=False,
                                escalate_on_health=False, lot=lot)
        gated = inspector.inspect(drifted_img, category, method, float(threshold), float(band), compare=False, lot=lot)
        sev = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        sweep = [inspector.inspect(apply_drift(image, kind, s), category, method, compare=False,
                                   escalate_on_health=False, lot=lot)["score"] for s in sev]
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

    # ------------------------------------------------------ re-qualification
    def requal_views(category):
        req = store.csv("requalification.csv")
        dist = store.json("requalification_scores.json").get(category, {})
        if req.empty or category not in set(req.category):
            return charts.empty("Run the pipeline stage 'requalify' to see this product"), charts.empty("—")
        r = req[req.category == category].iloc[0]
        if not bool(r.lot_shift):
            message = charts.empty(f"No lot change detected for {category} (p = {r.p_value:.2f}) — the model qualified on the "
                                   "training lot stays deployed. Select Carpet to see a re-qualification.", 340)
            groups = {"current-lot good parts (training-lot model)": np.asarray(dist.get("current_lot_good_base", []))}
            pol = inspector.policy(category, "patchcore")
            check = (charts.score_distribution(groups, pol["threshold"], pol["band"],
                                               f"Lot-shift check for {category}: current-lot good parts stay in PASS")
                     if len(groups[next(iter(groups))]) else charts.empty("No distributions"))
            return message, check
        before = {k: r[f"base_{k}"] for k in ("false_reject_rate", "escape_rate", "good_check_rate")}
        after = {k: r[f"requal_{k}"] for k in ("false_reject_rate", "escape_rate", "good_check_rate")}
        bars = charts.requal_chart(before, after, f"{category}: training-lot model vs re-qualified (operating split)")
        groups = {}
        if dist:
            groups = {"current-lot good parts (scored by the training-lot model)": dist.get("current_lot_good_base", []),
                      "current-lot good parts (re-qualified, leave-one-out)": dist.get("good_loo", []),
                      "synthetic defects made from the current lot": dist.get("synthetic", [])}
        dist_fig = (charts.score_distribution({k: np.asarray(v) for k, v in groups.items() if len(v)},
                                              float(dist.get("threshold", 50)), float(dist.get("band", 0)),
                                              f"Re-qualification of {category}: score distributions")
                    if groups else charts.empty("No distributions"))
        return bars, dist_fig

    def load_requal_samples(category):
        folder = store.examples / "requalify" / category
        files = sorted(str(p) for p in folder.glob("*.png")) if folder.exists() else []
        if not files:
            raise gr.Error("No sample current-lot images shipped for this product.")
        return files

    def run_requalify(files, category, session_lots):
        if not files or len(files) < 3:
            raise gr.Error("Upload at least 3 GOOD parts of the current lot (10+ recommended), or load the samples.")
        paths = [f if isinstance(f, str) else getattr(f, "name", str(f)) for f in files]
        started = time.perf_counter()
        lot = inspector.requalify(category, [read_rgb(p) for p in paths])
        seconds = time.perf_counter() - started
        session_lots = dict(session_lots or {})
        session_lots[category] = lot
        shift = lot["lot_shift"]
        tiles = _stat_tiles([(str(lot["n_good"]), "good parts of the current lot", None),
                             (f"{shift['current_lot_median']:.1f} vs {shift['qualification_median']:.1f}",
                              "median score: current lot vs qualification", None),
                             (f"{shift['p_value']:.3g}", "lot-shift test p-value (< 0.01 = changed)",
                              "CHECK" if shift["shifted"] else "PASS"),
                             (f"{lot['threshold']:.1f} ± {lot['band']:.1f}", "new threshold ± CHECK band", None),
                             (f"{lot['added_patches']:,}", "new patches added to the memory bank", None),
                             (f"{seconds:.1f} s", "time to re-qualify (CPU)", None)])
        dist = charts.score_distribution({"current-lot good parts (leave-one-out)": np.asarray(lot["good_scores"]),
                                          "synthetic defects made from them": np.asarray(lot["synthetic_scores"])},
                                         lot["threshold"], lot["band"], "Your re-qualification (no real defects)")
        gr.Info("Re-qualified. The Live demo now uses this model when 'Current lot (re-qualified)' is selected.")
        return (session_lots, tiles, dist, lot["synthetic_examples"], lot_radio(category, session_lots, LOT_CURRENT),
                lot["threshold"], lot["band"])

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
        lot = inspector.shipped_lot(category) if method == "patchcore" else None  # deployed: current lot if re-qualified
        r = inspector.inspect(image, category, method, threshold, band, lot=lot)
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
        lot = inspector.shipped_lot(category) if method == "patchcore" else None
        frame, _ = inspector.inspect_batch([read_rgb(p) for p in paths], [Path(p).name for p in paths], category, method,
                                           lot=lot)
        path = Path(tempfile.mkdtemp(prefix="ds_api_")) / "batch.csv"
        frame.to_csv(path, index=False)
        return frame.to_dict(orient="records"), str(path)

    def health():
        # Gradio clients treat any dict with a "path" key as a file to download, so the manifest's
        # "path" is exposed as "file" instead.
        extractor = {("file" if k == "path" else k): v
                     for k, v in (inspector.manifest.get("feature_extractor") or {}).items()}
        return {"status": "ok", "version": inspector.manifest.get("version"), "categories": categories,
                "methods": {c: inspector.methods(c) for c in categories},
                "feature_extractor": extractor,
                "policies": {c: {m: inspector.policy(c, m) for m in inspector.methods(c)} for c in categories},
                "current_lot_requalified": {c: inspector.shipped_lot(c) is not None for c in categories}}

    # ------------------------------------------------------------ results data
    def confusion_figures() -> list:
        figures = []
        scores = store.json("scores.json")
        req_scores = store.json("requalification_scores.json")
        req = store.csv("requalification.csv")
        for category in categories:
            deployed_lot = (not req.empty and category in set(req.category)
                            and bool(req[req.category == category].iloc[0].deployed))
            if deployed_lot and category in req_scores:
                d = req_scores[category]
                counts = charts.confusion_counts(d["operating_scores_requalified"], d["operating_labels"],
                                                 d["threshold"], d["band"])
                title = f"{category_labels[category]} · current lot (re-qualified)"
            elif category in scores:
                data = scores[category]
                op = np.asarray(data["test_roles"]) == "operating_test"
                s = np.asarray(data["methods"]["patchcore"]["test"])[op]
                lab = np.asarray(data["test_labels"])[op]
                pol = inspector.policy(category, "patchcore")
                counts = charts.confusion_counts(s, lab, pol["threshold"], pol["band"])
                title = f"{category_labels[category]} · deployed PatchCore"
            else:
                figures.append(charts.empty("No scores"))
                continue
            figures.append(charts.confusion_heatmap(counts, title, 300))
        return figures

    # ================================================================ layout
    blocks_kwargs = {"title": "DefectSpotter · visual inspection without defect labels"}
    if GRADIO_MAJOR < 6:
        blocks_kwargs.update(theme=build_theme(), css=CSS, js=FORCE_LIGHT_JS)
    initial_lot = LOT_CURRENT if lot_available(default_category, None) else LOT_TRAIN
    initial_policy = current_policy(default_category, default_method, initial_lot, None)
    with gr.Blocks(**blocks_kwargs) as demo:
        queue_state = gr.State([])
        audit_state = gr.State([])
        example_state = gr.State(None)
        last_state = gr.State(None)
        onboard_state = gr.State(None)
        lots_state = gr.State({})

        gr.HTML(ex.hero(store, inspector))
        gr.HTML(ex.nav())
        with gr.Group(elem_classes="ds-settings"):
            gr.Markdown(ex.settings_help(), elem_classes="ds-note")
            with gr.Row(equal_height=True):
                category = gr.Dropdown(_labelled(categories, category_labels), value=default_category,
                                       label="Product", scale=2, info="which production line")
                method = gr.Dropdown(_labelled(inspector.methods(default_category), METHOD_LABELS), value=default_method,
                                     label="Detector", scale=3, info="PatchCore is the deployed detector")
                lot_choice = gr.Radio([LOT_TRAIN, LOT_CURRENT] if initial_lot == LOT_CURRENT else [LOT_TRAIN],
                                      value=initial_lot, label="Production lot", scale=3,
                                      info="re-qualified models exist where a lot change was detected")
                threshold = gr.Slider(0, 100, value=initial_policy["threshold"], step=0.5, label="Decision threshold", scale=3,
                                      info="lower = fewer escapes, more false rejects")
                band = gr.Slider(0, 20, value=initial_policy["band"], step=0.5, label="CHECK band ±", scale=2,
                                 info="near the threshold → human")
                reset = gr.Button("↺ Calibrated", scale=1, size="sm")
            policy_md = gr.Markdown(policy_text(default_category, default_method, initial_lot, None), elem_classes="ds-note")

        with gr.Tabs():
            # ======================================================= ① overview
            with gr.Tab("🏠 Overview"):
                gr.HTML(ex.intro("🏠", "Overview — the problem, the solution, the results",
                                 "A quality inspector at the end of a production line must stop defective parts without "
                                 "scrapping good ones — but defects are rare, so there are almost no defect images to "
                                 "learn from. DefectSpotter learns only from good parts.",
                                 "Read top to bottom: problem → how it works → headline results → economics → responsible "
                                 "use. The tabs to the right go deeper; the <b>🎬 Live demo</b> is the last tab."))
                gr.HTML(problem_html())
                gr.HTML(pipeline_html())
                gr.HTML(results_html(store))
                ops_table = ex.deployed_ops(store, inspector)
                if not ops_table.empty and "false_reject_rate" in ops_table:
                    shown = ops_table.copy()
                    for col in ("false_reject_rate", "escape_rate", "good_check_rate", "review_rate"):
                        shown[col] = (100 * shown[col]).round(1).astype(str) + "%"
                    shown["cost"] = shown["cost"].round(0)
                    gr.Dataframe(shown.rename(columns={"category": "product", "lot": "model", "false_reject_rate": "false rejects",
                                                       "escape_rate": "escapes", "good_check_rate": "good parts → CHECK",
                                                       "review_rate": "all parts → CHECK", "cost": "cost / 1,000 parts"}),
                                 label="Deployed operating point per product (operating test split, never used for any choice)",
                                 interactive=False)
                with gr.Row():
                    with gr.Column():
                        gr.Plot(charts.threshold_dumbbell(store.csv("threshold_comparison.csv"), "patchcore"), show_label=False)
                        _cap("<b>Can we set the threshold without real defects?</b> Each row compares the threshold chosen "
                             "from synthetic defects (circles) with the one a plant would choose if it already had labelled "
                             "defects (diamond). Close together = the label-free method works.")
                    with gr.Column():
                        cost_model = store.json("cost_model.json")
                        cs = cost_summary(store)
                        gr.Plot(charts.cost_comparison(cost_model, cs["ours"]) if cost_model and np.isfinite(cs["ours"])
                                else charts.empty("No cost model"), show_label=False)
                        _cap("<b>Business case.</b> Expected cost per 1,000 parts (illustrative units: escape 400, false "
                             "reject 2, human check 0.30, 1% defects) for no inspection, 100% manual inspection, and "
                             "DefectSpotter with a human on CHECK.")
                gr.HTML(responsible_html())

            # =========================================================== ② data
            with gr.Tab("📊 Data"):
                comp = store.csv("dataset_composition.csv")
                stats = store.csv("image_stats.csv")
                gr.HTML(ex.intro("📊", "Data — where it comes from and what it looks like",
                                 "MVTec AD industrial images: good parts for training, real defects with pixel masks for "
                                 "testing. This page shows the source, the leakage-free split, exploratory data analysis "
                                 "(EDA) and the synthetic defects that replace real ones during calibration.",
                                 "Charts are interactive (hover, zoom, click legend entries to hide series). Galleries "
                                 "open full-size when clicked."))
                gr.HTML(ex.data_source_html(store))
                gr.HTML(ex.splits_html(store))
                _section("Composition and defect types")
                with gr.Row():
                    with gr.Column():
                        gr.Plot(charts.composition_chart(comp), show_label=False)
                        _cap("Images per product and split. Training = good only; the test set is defect-heavy "
                             "(≈70% defective) unlike a real line (≈1%), so rates are reported per class.")
                    with gr.Column():
                        eda_types = gr.Plot(charts.defect_types_chart(comp, default_category), show_label=False)
                        _cap("Defect types in the test set of the product selected above (change it in the global settings).")
                gr.Gallery([(p, f"{c}: good parts and every defect type (red = ground truth)") for c in
                            stats.category.unique() if (p := store.figure(f"eda_montage_{c}.jpg"))] if not stats.empty else [],
                           label="What the products and their defects look like", columns=2, height=430, object_fit="contain")
                _section("Acquisition and defect geometry")
                with gr.Row():
                    signal = gr.Dropdown(["brightness", "contrast", "sharpness", "saturation_pct", "r_mean", "g_mean", "b_mean"],
                                         value="brightness", label="Acquisition signal", scale=1,
                                         info="per-image statistic used by the camera-health gate")
                with gr.Row():
                    with gr.Column():
                        eda_box = gr.Plot(charts.acquisition_box(stats, "brightness"), show_label=False)
                        _cap("Spread of the chosen acquisition signal per product and split. Tight boxes = a controlled "
                             "camera — what makes good-only models work, and why drift matters.")
                    with gr.Column():
                        gr.Plot(charts.mask_area_chart(stats), show_label=False)
                        _cap("How large the real defects are (% of the image covered by the ground-truth mask). Many are "
                             "tiny — why localisation metrics are reported next to image AUROC.")
                with gr.Row():
                    gr.Gallery([(p, f"where defects occur · {c}") for c in (stats.category.unique() if not stats.empty else [])
                                if (p := store.figure(f"eda_defect_location_{c}.png"))],
                               label="Defect location heat maps (sum of all ground-truth masks; bright = frequent)", columns=4, height=260)
                    gr.Gallery([(p, f"mean | std · {c}") for c in categories if (p := store.figure(f"baseline_mean_std_{c}.jpg"))],
                               label="Mean good image and per-pixel variability (classical baseline)", columns=3, height=260)
                _section("Synthetic defects (replace real defects during calibration)")
                gr.HTML(ex.synthetic_html())
                gr.Gallery(store.synthetic_gallery() + store.diffusion_gallery(),
                           label="Procedural (cut-paste · scratch · stain) and Stable-Diffusion-refined synthetic defects",
                           columns=6, height=300)
                gr.Markdown(_eda_insights(stats, comp))
                with gr.Accordion("Table: dataset composition", open=False):
                    gr.Dataframe(comp, interactive=False)

            # ================================================ ③ models & training
            with gr.Tab("🧠 Models & training"):
                gr.HTML(ex.intro("🧠", "Models & training — what each component is and how it was trained",
                                 "One frozen deep backbone, a memory of normal patches (PatchCore), two generative VAEs and a "
                                 "classical baseline. Nothing is ever trained on defects.",
                                 "Start with the table <i>What is trained, and on what?</i> — then one card per component, "
                                 "with the exact parameters read from the shipped models."))
                gr.HTML(ex.components_html(inspector, store))
                gr.HTML(ex.backbone_html(inspector))
                with gr.Row():
                    with gr.Column():
                        gr.Plot(charts.backbone_chart(store.csv("backbone_study.csv")), show_label=False)
                        _cap("<b>Backbone study.</b> Accuracy vs size of candidate backbones (PatchCore on each). ResNet-18 "
                             "gives the best accuracy within the 10–20 MB deployment budget.")
                    with gr.Column():
                        gr.Plot(charts.footprint_chart(store.csv("artifact_sizes.csv")), show_label=False)
                        _cap("<b>Deployment footprint.</b> Size of every shipped model file (ONNX backbone, memory banks, VAEs).")
                gr.HTML(ex.patchcore_html(inspector))
                hpo = store.csv("hpo_patchcore.csv")
                gr.Dataframe(_recipe_table(hpo), interactive=False,
                             label="Deployed PatchCore recipe per product (chosen on held-out good + synthetic defects only)")
                gr.Dataframe(_ablation_table(hpo), interactive=False,
                             label="What each design choice is worth: best label-free objective per option (differences < 0.005 are noise)")
                gr.HTML(ex.vae_html(inspector, store))
                with gr.Row():
                    with gr.Column():
                        gr.Plot(charts.vae_history_chart(store.csv("fvae_training_history.csv"),
                                                         "Feature-space VAE training on good parts only",
                                                         "validation loss (feature MSE)"), show_label=False)
                        _cap("Validation loss per epoch (held-out good feature maps). The checkpoint with the lowest "
                             "validation loss is kept.")
                    with gr.Column():
                        gr.Plot(charts.vae_history_chart(store.csv("vae_training_history.csv")), show_label=False)
                        _cap("Pixel VAE (v1): validation L1 + SSIM loss per epoch on held-out good images.")
                gr.HTML(ex.baseline_html())
                gr.HTML(ex.calibration_html(inspector))
                with gr.Accordion("Tables: full hyper-parameter searches", open=False):
                    gr.Dataframe((hpo.sort_values(["category", "objective"], ascending=[True, False]) if "objective" in hpo
                                  else hpo).round(4), label="PatchCore search (synthetic validation only, best first)")
                    gr.Dataframe(store.csv("hpo_mean_baseline.csv").round(4), label="Baseline alignment search")
                    gr.Dataframe(store.csv("backbone_study.csv").round(4), label="Backbone study (test split, analysis only)")

            # ======================================================== ④ results
            with gr.Tab("📈 Results"):
                gr.HTML(ex.intro("📈", "Results on held-out real defects",
                                 "Everything here is measured on MVTec test images that never trained a model or chose a "
                                 "threshold. First what the metrics mean, then the confusion matrices of the deployed "
                                 "decisions, then ranking quality, localisation and speed.",
                                 "Product-specific charts follow the <b>Product</b> selected in the global settings."))
                gr.HTML(ex.glossary_html())
                _section("Decisions of the deployed system (confusion matrices, operating test split)")
                with gr.Row():
                    for fig in confusion_figures():
                        gr.Plot(fig, show_label=False)
                _cap("Rows = ground truth, columns = decision. Each cell shows the count and the share of its row. Ideal: "
                     "good parts in ✓ PASS, defective parts in ✕ REJECT; ⚠ CHECK goes to an inspector. Carpet uses the "
                     "model re-qualified for the current lot (see Drift & new lots).")
                _section("Ranking quality (threshold-independent)")
                with gr.Row():
                    metric_choice = gr.Dropdown([("Image AUROC", "image_auroc"), ("Pixel AUROC", "pixel_auroc"),
                                                 ("AUPRO @ 30% FPR", "aupro_30"), ("Average precision", "image_ap")],
                                                value="image_auroc", label="Metric", scale=1)
                with gr.Row():
                    with gr.Column():
                        metric_plot = gr.Plot(charts.metric_bars(metrics, "image_auroc", "Image AUROC"), show_label=False)
                        _cap("Bars per product and detector; whiskers = 95% bootstrap confidence interval.")
                    with gr.Column():
                        roc_plot = gr.Plot(charts.roc_chart(store.json("roc_curves.json"), default_category), show_label=False)
                        _cap("ROC curve: share of defects caught (y) vs share of good parts flagged (x) as the threshold "
                             "moves. Top-left corner = perfect; the diagonal = guessing.")
                with gr.Row():
                    with gr.Column():
                        gr.Plot(charts.auroc_heatmap(metrics), show_label=False)
                        _cap("<b>Heat map</b> of image AUROC for every product × detector: the feature-based detectors "
                             "(PatchCore, feature-space VAE) dominate the pixel VAE and the classical baseline.")
                    with gr.Column():
                        gr.Plot(charts.accuracy_latency(metrics, store.csv("latency.csv")), show_label=False)
                        _cap("Accuracy vs speed per product and detector — top-left is best.")
                _section("Per defect type and localisation")
                with gr.Row():
                    with gr.Column():
                        defect_plot = gr.Plot(charts.defect_type_bars(store.csv("per_defect_type.csv"), default_category,
                                                                      default_method), show_label=False)
                        _cap("Decision mix per defect type at the calibrated policy — which defects are caught, sent to "
                             "a human, or missed.")
                    with gr.Column():
                        scatter_plot = gr.Plot(charts.method_scatter(store.json("scores.json"), default_category), show_label=False)
                        _cap("Each dot is a test part: PatchCore score vs feature-space VAE score. Points far from the "
                             "diagonal show where the two methods disagree.")
                qualitative = gr.Image(store.figure(f"qualitative_{default_category}.jpg"),
                                       label="Heat maps on representative test parts (median-scoring example per defect type): "
                                             "input | ground truth | each detector", interactive=False, height=560)
                _section("Thresholds, speed")
                with gr.Row():
                    with gr.Column():
                        dumbbell = gr.Plot(charts.threshold_dumbbell(store.csv("threshold_comparison.csv"), default_method),
                                           show_label=False)
                        _cap("Thresholds of the selected detector: synthetic-defect policies vs the real-defect oracle.")
                    with gr.Column():
                        gr.Plot(charts.latency_bars(store.csv("latency.csv")), show_label=False)
                        _cap("Median end-to-end time per part on a 4-core CPU (decode → features → decision).")
                with gr.Accordion("Tables: all metrics · thresholds · latency", open=False):
                    gr.Dataframe(metrics.round(4) if not metrics.empty else metrics,
                                 label="All metrics (test split; CI = 95% stratified bootstrap)", interactive=False)
                    gr.Dataframe(store.csv("threshold_comparison.csv").round(2), label="Synthetic vs real-defect thresholds")
                    gr.Dataframe(store.csv("latency.csv").round(1), label="Latency")

            # ================================================= ⑤ threshold & cost
            with gr.Tab("🎚️ Threshold & cost"):
                gr.HTML(ex.intro("🎚️", "Threshold & cost — the trade-off priced in money",
                                 "Moving the threshold trades false rejects (scrapped good parts) against escapes (defects "
                                 "reaching the customer). Set the economics of your line and see the expected cost move; "
                                 "<b>Apply</b> recomputes the cost-optimal threshold from good + synthetic parts only.",
                                 "Use the global threshold and CHECK-band sliders at the top; all charts update."))
                with gr.Row():
                    prevalence = gr.Slider(0.1, 10, value=1.0, step=0.1, label="Defect rate (%)", info="share of defective parts")
                    c_escape = gr.Number(400, label="Cost of an escape (per part)")
                    c_fr = gr.Number(2, label="Cost of a false reject (per part)")
                    c_review = gr.Number(0.30, label="Cost of a human CHECK (per part)")
                    capacity = gr.Slider(1, 30, value=10, step=1, label="Inspector capacity (% of parts)",
                                         info="planned at ≤ 50% (headroom for drift)")
                    per_day = gr.Number(20000, label="Parts per day", precision=0)
                with gr.Row():
                    eval_set = gr.Radio(["Operating test split (real defects)", "Calibration (good + synthetic defects)"],
                                        value="Operating test split (real defects)", label="Evaluate on", scale=3)
                    apply_btn = gr.Button("⚙ Apply cost-optimal threshold (no real defects used)", variant="primary", scale=2)
                lab_tiles = gr.HTML()
                lab_dist = gr.Plot(show_label=False)
                _cap("Score distributions of good parts, real defects and synthetic defects with the PASS / CHECK / REJECT "
                     "zones of your current threshold.")
                with gr.Row():
                    with gr.Column():
                        lab_rates = gr.Plot(show_label=False)
                        _cap("False-reject, escape and review rates as the threshold moves (your CHECK band fixed).")
                    with gr.Column():
                        lab_cost = gr.Plot(show_label=False)
                        _cap("Expected cost per 1,000 parts vs threshold; the marker is the cost optimum.")
                with gr.Row():
                    with gr.Column():
                        lab_matrix = gr.Plot(show_label=False)
                        _cap("Confusion matrix at your threshold on the chosen evaluation set.")
                    with gr.Column():
                        lab_policies = gr.Dataframe(label="Threshold policies compared on the operating test split",
                                                    interactive=False)
                gr.Markdown("**Label-free detection capability.** Before a synthetic set may influence the default "
                            "threshold, the detector must separate it from held-out good parts (synthetic AUROC ≥ 0.9). "
                            "Sets below that are the system's documented detection limit — like a measurement-capability "
                            "study on a real line.")
                lab_capability = gr.Dataframe(label="Synthetic AUROC by calibration set (no real defects involved)",
                                              interactive=False)

            # ================================================ ⑥ drift & new lots
            with gr.Tab("🌗 Drift & new lots"):
                gr.HTML(ex.intro("🌗", "Drift & new lots — keeping the model honest after deployment",
                                 "A good-only model calls anything unusual a defect — including a dimmer lamp, a defocused "
                                 "lens, a bumped camera or a new production lot. This page shows how each is detected and "
                                 "handled: the camera-health gate, control charts, and re-qualification for new lots.",
                                 "Section A simulates a change on one part, B replays a 200-part production stream, C "
                                 "re-qualifies a product for a new lot (try it with the 10 sample parts)."))
                _section("A · Simulate an acquisition change on one part")
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
                    with gr.Column():
                        drift_sweep = gr.Plot(show_label=False)
                        _cap("Score of this part as the change gets stronger — it climbs into CHECK/REJECT although the "
                             "part is unchanged. The camera-health gate catches it first.")
                    drift_health = gr.HTML()
                _section("B · Production stream replay (200 parts, ~3% defects)")
                stream_plot = gr.Plot(show_label=False)
                _cap("Anomaly score per part with its EWMA control chart: parts 1–60 qualified, 61–120 the lamp ages, "
                     "121–150 after maintenance, 151+ the camera is bumped. Red markers = sustained alarms.")
                psi_plot = gr.Plot(show_label=False)
                _cap("Population stability index (PSI) of the score distribution in a rolling window vs qualification: "
                     "< 0.1 stable, 0.1–0.25 watch, > 0.25 alarm.")
                with gr.Row():
                    bright_plot = gr.Plot(show_label=False)
                    sharp_plot = gr.Plot(show_label=False)
                alarm_table = gr.Dataframe(label="Part number of the first sustained alarm (2 in a row) after each change",
                                           interactive=False)
                with gr.Row():
                    sens_frr = gr.Plot(show_label=False)
                    sens_flag = gr.Plot(show_label=False)
                    sens_escape = gr.Plot(show_label=False)
                _cap("<b>Drift heat maps</b> (rows = kind of change, columns = severity): left — good parts falsely "
                     "rejected; middle — share caught by the camera-health gate; right — real defects that escape.")
                gr.Markdown(MONITORING_POLICY)
                _section("C · New production lot → re-qualification")
                gr.HTML(ex.lot_html(store))
                with gr.Row():
                    with gr.Column():
                        requal_bars = gr.Plot(show_label=False)
                        _cap("Effect of re-qualification on the operating test split: good parts sent to CHECK drop "
                             "sharply while escapes stay low and false rejects stay at zero.")
                    with gr.Column():
                        requal_dist = gr.Plot(show_label=False)
                        _cap("Current-lot good parts scored by the old model (shifted right) vs after re-qualification, "
                             "and the synthetic defects made from them that set the new threshold.")
                gr.Markdown("**Try it:** load the good parts of the current lot (or upload your own — good parts only) and "
                            "press **Re-qualify**. The new model is used by the Live demo whenever *Production lot* is set "
                            "to *Current lot (re-qualified)*.")
                with gr.Row():
                    requal_files = gr.File(file_count="multiple", file_types=["image"], label="Good parts of the current lot", scale=3)
                    with gr.Column(scale=1):
                        requal_samples = gr.Button("📂 Load the sample current-lot good parts")
                        requal_btn = gr.Button("🔁 Re-qualify for the current lot", variant="primary")
                requal_tiles = gr.HTML()
                with gr.Row():
                    requal_live_dist = gr.Plot(show_label=False)
                    requal_gallery = gr.Gallery(label="Synthetic defects generated from the current lot", columns=3, height=300)

            # ===================================================== ⑦ new product
            with gr.Tab("🆕 New product"):
                gr.HTML(ex.intro("🆕", "New product — onboarding from a handful of good parts",
                                 "PatchCore needs no network training, so a new product line can be onboarded in seconds "
                                 "from a few good images: the memory bank is built from them, synthetic defects calibrate "
                                 "the threshold. Held-out experiment: metal nut.",
                                 "The curve shows accuracy vs number of good images; below, onboard the metal nut live."))
                with gr.Row():
                    with gr.Column():
                        gr.Plot(charts.few_shot_chart(store.csv("few_shot.csv")), show_label=False)
                        _cap("Image AUROC on real metal-nut defects vs the number of good images in the memory bank "
                             "(mean over random draws). 8 good images already give ≈ 0.98.")
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

            # ========================================================== ⑧ guide
            with gr.Tab("📖 Guide"):
                gr.HTML(ex.intro("📖", "Guide — every concept, model and parameter explained",
                                 "The complete project guide (also in the repository as <code>docs/GUIDE.md</code>): what "
                                 "each model is, why it was chosen, how it was trained and how to run everything.",
                                 None))
                gr.HTML(ex.glossary_html())
                gr.Markdown(guide_path.read_text(encoding="utf-8") if guide_path.exists()
                            else "_docs/GUIDE.md not found in this build._")

            # ===================================================== 🎬 live demo
            with gr.Tab("🎬 Live demo"):
                gr.HTML(ex.intro("🎬", "Live demo — inspect parts like an inspector at the end of the line",
                                 "Upload or photograph a part (or click a held-out sample) and DefectSpotter returns a score, "
                                 "a heat map of the suspected defect and a PASS / CHECK / REJECT decision. Batches and the "
                                 "human review queue are in the sub-tabs.",
                                 "The decision card explains <i>why</i>; the score bar shows where the part falls relative to "
                                 "the threshold; the heat map shows <i>where</i>. Demo images to upload are in "
                                 "<code>demo_images/</code> of the repository."))
                gr.HTML(ex.demo_script_html())
                with gr.Tabs():
                    with gr.Tab("🔍 Inspect a part"):
                        with gr.Row():
                            with gr.Column(scale=5):
                                image = gr.Image(type="numpy", image_mode="RGB", sources=["upload", "webcam", "clipboard"],
                                                 label="Part image · upload, paste or take a photo", height=360)
                                with gr.Row():
                                    inspect_btn = gr.Button("🔍 Inspect part", variant="primary")
                                    queue_btn = gr.Button("➕ Send to review queue")
                                examples = gr.Gallery(value=store.example_gallery(default_category),
                                                      label="Held-out test samples of the selected product (click to "
                                                            "inspect · ground truth is revealed after)",
                                                      columns=5, height=270, allow_preview=False, object_fit="cover")
                            with gr.Column(scale=6):
                                card = gr.HTML(decision_card("PASS", 0, initial_policy["threshold"], initial_policy["band"],
                                                             "waiting for an image", 0).replace("✓ PASS", "Ready").replace(
                                    "Within the learned envelope of good parts — ship.", "Inspect a part to get a decision."))
                                bar = gr.Plot(charts.decision_bar(0, initial_policy["threshold"], initial_policy["band"]),
                                              show_label=False)
                                with gr.Row():
                                    overlay_img = gr.Image(label="Heat map + suspected regions (boxes)", height=300,
                                                           interactive=False)
                                    truth_img = gr.Image(label="Input (blue outline = ground-truth defect for samples)",
                                                         height=300, interactive=False)
                        explanation = gr.Markdown()
                        with gr.Row():
                            with gr.Column():
                                compare = gr.Gallery(label="All detectors on this part · score · decision", columns=5,
                                                     height=250, allow_preview=True, object_fit="contain")
                                _cap("The same part through every detector: memory bank (PatchCore), feature-space VAE, "
                                     "pixel VAE, mean-image baseline and the ensemble — each with its own calibrated threshold.")
                            with gr.Column():
                                recon = gr.Image(label="Pixel VAE: input (left) vs its reconstruction of "
                                                       "'what a good part looks like' (right)", height=250, interactive=False)
                                _cap("Where input and reconstruction differ, the VAE suspects a defect. Blurry everywhere = "
                                     "why the feature-space VAE replaced it as the generative detector.")
                        with gr.Row():
                            with gr.Column():
                                heat = gr.Plot(charts.anomaly_heatmap(None), show_label=False)
                                _cap("Interactive pixel anomaly map (0–100). Above 50 = more unusual than 99.9% of good-part "
                                     "pixels; the contour marks suspected regions.")
                            with gr.Column():
                                regions = gr.Dataframe(label="Suspected regions (pixel score > 50)", interactive=False)
                                health_box = gr.HTML()
                        with gr.Accordion("📷 Live camera — continuous inspection at ~1 frame per second", open=False):
                            with gr.Row():
                                live_cam = gr.Image(sources=["webcam"], streaming=True, type="numpy", label="Webcam", height=320)
                                live_overlay = gr.Image(label="Live heat map", height=320, interactive=False)
                                live_card = gr.HTML()
                        with gr.Accordion("Machine-readable inspection record (same as the /inspect_part API)", open=False):
                            record = gr.JSON()

                    with gr.Tab("📦 Batch"):
                        gr.Markdown("Inspect a whole tray or shift at once. CHECK decisions are routed automatically to "
                                    "the **Review queue**; the decision table can be downloaded as CSV for the MES/QMS.")
                        with gr.Row():
                            batch_files = gr.File(file_count="multiple", file_types=["image"], label="Part images", scale=3)
                            with gr.Column(scale=1):
                                sample_btn = gr.Button("📂 Load sample batch for this product")
                                batch_btn = gr.Button("▶ Inspect batch", variant="primary")
                        batch_tiles = gr.HTML()
                        batch_chart = gr.Plot(charts.batch_strip(pd.DataFrame(), 50, 5), show_label=False)
                        _cap("Every part of the batch on the 0–100 score scale against the PASS / CHECK / REJECT zones.")
                        batch_table = gr.Dataframe(label="Batch summary", interactive=False)
                        with gr.Row():
                            batch_gallery = gr.Gallery(label="Overlays", columns=6, height=320, scale=4)
                            batch_csv = gr.File(label="Download batch CSV", scale=1)

                    with gr.Tab("🧑‍🔧 Review queue"):
                        gr.Markdown("**Human in the loop.** Every CHECK decision waits here for a trained inspector. Your "
                                    "verdict, note, the model score, threshold and detector are written to an audit log — "
                                    "the model is never silently retrained from these decisions.")
                        queue_tiles = gr.HTML(_queue_stats([]))
                        queue_table = gr.Dataframe(queue_frame([]), label="Queue", interactive=False)
                        with gr.Row():
                            with gr.Column(scale=2):
                                pending = gr.Dropdown([], label="Pending item")
                                note = gr.Textbox(label="Inspector note",
                                                  placeholder="e.g. scratch 3 mm on rim, confirmed under loupe")
                                with gr.Row():
                                    confirm_btn = gr.Button("✕ Confirm defect → REJECT", variant="stop")
                                    release_btn = gr.Button("✓ Release as good → PASS")
                            review_img = gr.Image(label="Selected part", height=320, interactive=False, scale=2)
                        audit_table = gr.Dataframe(pd.DataFrame(columns=AUDIT_COLUMNS), label="Audit log", interactive=False)
                        audit_file = gr.File(label="Download audit log (CSV)")

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
        inspect_inputs = [image, category, method, threshold, band, example_state, lot_choice, lots_state]
        category.change(on_category, [category, method, lots_state],
                        [method, lot_choice, threshold, band, policy_md, examples, example_state], **PRIVATE)
        method.input(on_method_or_lot, [category, method, lot_choice, lots_state], [threshold, band, policy_md], **PRIVATE)
        lot_choice.input(on_method_or_lot, [category, method, lot_choice, lots_state], [threshold, band, policy_md], **PRIVATE)
        reset.click(reset_policy, [category, method, lot_choice, lots_state], [threshold, band], **PRIVATE)
        inspect_btn.click(run_inspect, inspect_inputs, inspect_outputs, **PRIVATE)
        image.input(lambda: None, None, example_state, **PRIVATE).then(run_inspect, inspect_inputs, inspect_outputs, **PRIVATE)
        examples.select(select_example, [category], [image, example_state], **PRIVATE).then(
            run_inspect, inspect_inputs, inspect_outputs, **PRIVATE)
        live_cam.stream(live_inspect, [live_cam, category, method, threshold, band, lot_choice, lots_state],
                        [live_overlay, live_card], stream_every=1.0, time_limit=180, concurrency_limit=2, **PRIVATE)
        queue_btn.click(send_last_to_queue, [queue_state, last_state], [queue_state, queue_table, pending, queue_tiles], **PRIVATE)

        sample_btn.click(load_sample_batch, [category], [batch_files], **PRIVATE)
        batch_btn.click(run_batch, [batch_files, category, method, threshold, band, queue_state, lot_choice, lots_state],
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
              simulate_drift, [category, method, drift_file, drift_kind, severity, threshold, band, lot_choice, lots_state],
              [drift_base, drift_img, drift_card_base, drift_card_raw, drift_card_gated, drift_health, drift_sweep],
              trigger_mode="always_last", **PRIVATE)
        gr.on([category.change, demo.load], requal_views, [category], [requal_bars, requal_dist], **PRIVATE)
        requal_samples.click(load_requal_samples, [category], [requal_files], **PRIVATE)
        requal_btn.click(run_requalify, [requal_files, category, lots_state],
                         [lots_state, requal_tiles, requal_live_dist, requal_gallery, lot_choice, threshold, band], **PRIVATE).then(
            lambda c, m, l, s: policy_text(c, m, l, s), [category, method, lot_choice, lots_state], [policy_md], **PRIVATE)

        onboard_samples.click(lambda: store.new_product_samples("good")[:8], None, [onboard_files], **PRIVATE)
        onboard_btn.click(onboard, [onboard_files], [onboard_state, onboard_summary, onboard_dist, onboard_gallery], **PRIVATE)
        new_samples.select(_select_path(store.new_product_samples("test")), None, [new_image], **PRIVATE).then(
            inspect_new, [new_image, onboard_state], [new_card, new_overlay], **PRIVATE)
        new_image.upload(inspect_new, [new_image, onboard_state], [new_card, new_overlay], **PRIVATE)
        health_btn.click(health, None, health_json, **({**PRIVATE, 'api_name': 'refresh_health'} if GRADIO_MAJOR >= 6 else PRIVATE))

        api_kwargs = {"api_visibility": "public"} if GRADIO_MAJOR >= 6 else {}
        api_btn.click(api_inspect, [api_image, api_category, api_method, api_threshold, api_band], [api_out, api_overlay],
                      api_name="inspect_part", **api_kwargs)
        api_batch_btn.click(api_batch, [api_files, api_category, api_method], [api_batch_out, api_batch_csv],
                            api_name="inspect_batch", **api_kwargs)
        demo.load(health, None, health_json, api_name="health", **api_kwargs)
    return demo


RECIPE_COLUMNS = ("category", "image_size", "align", "pool_size", "bank_rotations", "coreset_ratio", "blur_sigma",
                  "score_mode", "topk_fraction", "bank_vectors", "bank_mb", "synthetic_image_auroc", "subtle_image_auroc",
                  "objective")


def _deployed_rows(hpo: pd.DataFrame) -> pd.DataFrame:
    """Rows of the recipes actually deployed (parsimony rule), falling back to the best objective."""
    if "selected" in hpo:
        chosen = hpo[hpo.selected.astype(str).str.lower() == "true"]
        if chosen.category.nunique() == hpo.category.nunique():
            return chosen
    return hpo.loc[hpo.groupby("category").objective.idxmax()]


def _recipe_table(hpo: pd.DataFrame) -> pd.DataFrame:
    """The configuration deployed for every product (label-free, parsimony rule)."""
    if hpo.empty or "objective" not in hpo:
        return hpo
    best = _deployed_rows(hpo).copy()
    best["best_objective_any_recipe"] = best.category.map(hpo.groupby("category").objective.max())
    return best[[c for c in (*RECIPE_COLUMNS, "best_objective_any_recipe") if c in best]].round(4).reset_index(drop=True)


def _ablation_table(hpo: pd.DataFrame) -> pd.DataFrame:
    """Best label-free objective per value of each v2/v3 design option."""
    options = {"image_size": ("input resolution", lambda v: f"{int(v)} px"),
               "align": ("pose normalisation", lambda v: "aligned" if str(v).lower() == "true" else "as captured"),
               "pool_size": ("feature aggregation", lambda v: "3×3 average" if int(v) == 3 else "raw descriptors"),
               "bank_rotations": ("good parts in the bank", lambda v: "as captured" if int(v) == 1 else f"{int(v)} orientations (90° steps)")}
    if hpo.empty or "objective" not in hpo:
        return pd.DataFrame()
    deployed = _deployed_rows(hpo).set_index("category")
    rows = []
    for option, (label, fmt) in options.items():
        if option not in hpo:
            continue
        for category, group in hpo.groupby("category"):
            best = group.groupby(group[option].astype(str)).objective.max().sort_values(ascending=False)
            if len(best) < 2:
                continue
            rows.append({"product": category, "design choice": label,
                         "best objective per option": " · ".join(f"{fmt(k)}: {v:.4f}" for k, v in best.items()),
                         "deployed": fmt(deployed.loc[category, option]) if category in deployed.index else "—"})
    return pd.DataFrame(rows)


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
                     f"(median {defects.mask_area_pct.median():.2f}%). Down-scaling is therefore a real risk, which is why "
                     "localisation metrics (pixel AUROC, AUPRO) are reported next to image AUROC.")
    for category, part in stats.groupby("category"):
        lines.append(f"- **{category}:** {int(part.width.iloc[0])}×{int(part.height.iloc[0])} px, brightness "
                     f"{part.brightness.mean():.0f}±{part.brightness.std():.0f}, sharpness {part.sharpness.mean():.0f}±"
                     f"{part.sharpness.std():.0f}.")
    lines.append("- **Acquisition is tightly controlled** (small brightness spread within a product). That is what lets a "
                 "good-only model work — and why a lamp or focus change on the real line (Drift tab) is dangerous.")
    lines.append("- **Test captures differ slightly from training captures** (carpet test images are brighter and from a "
                 "different roll) — handled by the lot re-qualification in the Drift & new lots tab.")
    lines.append("- **Test sets are defect-heavy** (≈70% defective) unlike a real line (≈1%). We therefore report rates per class "
                 "and re-weight costs and review load to a 1% defect rate.")
    return "\n".join(lines)


MONITORING_POLICY = """
### Monitoring policy (what an operator would do)
| Signal | Alarm rule | Action |
|---|---|---|
| Camera health: brightness, contrast, focus | EWMA (λ = 0.2) outside ±3σ of the qualified images, 2 parts in a row | stop & check lamp/lens/camera mount, re-qualify with good parts |
| Anomaly-score stream | EWMA of scores above the upper control limit | treat as process **or** imaging drift: inspect recent CHECK/REJECTs |
| CHECK rate | > 2× the calibrated review load over a shift | inspector capacity alert; lot change? → lot-shift test and re-qualification |
| Any single image out of qualified range | per-part health gate (|z| > 4 vs qualified images) | no automatic decision: PASS/REJECT → CHECK — never auto-ship or auto-scrap on an unqualified image |

Drift never triggers silent retraining: a memory-bank refresh (re-qualification) is a controlled change with re-validation.
"""


def api_markdown(categories: list[str]) -> str:
    return f"""
### Use DefectSpotter from code
Three public endpoints (see **Use via API** in the page footer for auto-generated docs):

| endpoint | inputs | returns |
|---|---|---|
| `/inspect_part` | image, category ({', '.join(categories)}), method (patchcore · fvae · vae · mean · ensemble), threshold, band (empty = calibrated) | JSON decision record + heat-map overlay |
| `/inspect_batch` | list of image files, category, method | per-part JSON rows + CSV |
| `/health` | – | versions, categories, calibrated policies, which products are re-qualified for the current lot |

```python
from gradio_client import Client, handle_file
client = Client("bhkr0901/defectspotter")          # or "http://127.0.0.1:7860" locally
record, overlay = client.predict(handle_file("part.png"), "bottle", "patchcore", None, None, api_name="/inspect_part")
print(record["decision"], record["score"])
```
PatchCore requests use the current-lot model wherever the product was re-qualified (same default as the app).
"""


def launch_kwargs() -> dict:
    if GRADIO_MAJOR >= 6:
        return {"theme": build_theme(), "css": CSS, "js": FORCE_LIGHT_JS}
    return {}
