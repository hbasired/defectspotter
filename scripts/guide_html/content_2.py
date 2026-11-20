"""Guide chapters 11-27: decisions, operations, results, the app, running it, reference."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .helpers import (callout, chapter, code, details, e, fig, formula, kpis, pct, steps, sub, table, thumbs, zones)


# ===================================================================================================== 11. decision
def ch_decision(c) -> str:
    d = c.d
    cm = d.cost
    scale_rows, pix_rows, zone_html = [], [], []
    for cat in d.cats:
        info = d.cal[cat]["methods"]["patchcore"]
        sc, ps = info["score_scale"], info["pixel_scale"]
        scale_rows.append([cat, f"{sc['good_median']:.3f}", f"{sc['synthetic_median']:.3f}", f"{sc['spread']:.3f}"])
        pix_rows.append([cat, f"{ps['low']:.3f}", f"{ps['threshold']:.3f}", f"{ps['high']:.3f}"])
        dep = d.deployed(cat)
        zone_html.append(f"<p style='margin:14px 0 0'><b>{cat}</b> — {'re-qualified lot model' if dep['lot'] else 'qualification model'}: "
                         f"threshold {dep['threshold']:g}, band ±{dep['band']:g}</p>{zones(dep['threshold'], dep['band'])}")
    pol_names = ["synthetic_cost", "synthetic_plus_diffusion_cost", "diffusion_cost", "synthetic_conspicuous_cost",
                 "good_quantile", "real_oracle_cost"]
    pol_rows = []
    for name in pol_names:
        row = [f"<code>{name}</code>", c.policy_label(name)]
        for cat in d.cats:
            p = d.cal[cat]["methods"]["patchcore"]["policies"].get(name)
            row.append(f"{p['threshold']:.1f} ± {p['band']:g}" if p else "—")
        pol_rows.append(row)
    thr = d.thresholds[d.thresholds.method == "patchcore"]
    gap_rows = [[r.category, f"{r.synthetic_threshold:.1f}", f"{r.real_threshold:.1f}", f"{r.gap_synthetic_vs_real:+.1f}"]
                for r in thr.itertuples()]
    body = f"""
{sub("dec-scale", "11.1", "A common 0–100 score scale")}
<p>Raw scores mean nothing to an operator (a PatchCore distance of 2.7, a VAE error of 0.04 …). Every detector's score is
mapped linearly so that <b>25 = the median held-out good part</b> and <b>75 = the median synthetic defect</b>:</p>
{formula("score = 25 + 50 × (raw − median_good) / spread          spread = max( median_synthetic − median_good, 4 × std_good )")}
{table(["product (PatchCore)", "median raw score, good", "median raw score, synthetic", "spread"], scale_rows, num_cols=(1, 2, 3))}
<p>Heat maps get a similar, piecewise-linear <b>pixel scale</b>: 0 = median good pixel, <b>50 = more unusual than 99.9% of
good pixels</b>, 100 = the median pixel inside synthetic-defect masks. Suspected regions are the connected areas above 50.</p>
{table(["product (PatchCore)", "raw value → 0", "raw value → 50 (99.9% quantile)", "raw value → 100"], pix_rows, num_cols=(1, 2, 3))}
{sub("dec-zones", "11.2", "Three decisions from one threshold and a band")}
{formula("PASS    if score < t − b\nCHECK   if t − b ≤ score < t + b\nREJECT  if score ≥ t + b")}
{"".join(zone_html)}
{sub("dec-cost", "11.3", "Choosing t and b: the cost model")}
<p>Every candidate (t, b) is priced as an <b>expected cost per part</b> at a realistic defect rate:</p>
{formula('''cost per part = (1 − p) · FRR · C_false_reject  +  p · escape_rate · C_escape  +  review_rate · C_check
p = {cm['defect_prevalence']:g} (1% defective)    C_escape = {cm['cost_escape']:g}    C_false_reject = {cm['cost_false_reject']:g}    C_check = {cm['cost_review']:g}  (USD)''')}
<ul><li><b>Grid:</b> t from 0 to 100 in steps of 0.5; b ∈ {{0, 2.5, 5, 7.5, 10, 12.5, 15}}.</li>
<li><b>Smoothed rates.</b> With ≈ 50 good parts a raw false-reject rate moves in 2% steps and the cheapest threshold
sits right on top of one unlucky part. The rates are therefore read from <b>kernel-density estimates</b> of the score
distributions (Gaussian kernels, Silverman's bandwidth, floor 0.5 points) — smooth and stable.</li>
<li><b>Capacity.</b> The CHECK band may use at most <b>{100 * cm['review_capacity'] * cm['review_planning_share']:.0f}%</b> of the parts:
{100 * cm['review_capacity']:.0f}% inspector capacity × {cm['review_planning_share']:.1f} planning share — headroom for drift and new
lots, as in any capacity planning.</li>
<li><b>Capability.</b> Only defect sets the detector can see (AUROC ≥ 0.9, chapter 5.4) enter the defect side.</li></ul>
<p>The app's <b>Threshold &amp; cost</b> tab recomputes all of this live with your own numbers.</p>
{sub("dec-policies", "11.4", "All threshold policies that were computed (PatchCore)")}
{table(["policy", "defects used", *d.cats], pol_rows, caption="t ± b per product. The first row is deployed; the last uses "
       "real labelled defects (oracle split) and is computed only to measure how close the label-free choice lands.")}
{table(["product", "label-free threshold", "threshold with real labels", "gap"], gap_rows, num_cols=(1, 2, 3),
       caption="The label-free threshold lands within a few points of what a plant with labelled defects would choose.")}
{callout("code", "In the code", "<p><code>defectspotter/decision.py</code> — <code>fit_score_scale</code>, <code>fit_pixel_scale</code>, "
         "<code>decide</code>, <code>smoothed_rates</code>, <code>select_threshold_by_cost</code>, <code>calibration_sets</code>, "
         "<code>default_policy</code>.</p>")}"""
    return chapter("decision", 11, "From a score to a decision", body,
                   "How a raw distance becomes PASS, CHECK or REJECT — with thresholds chosen without a single real defect.")


# ===================================================================================================== 12. requal
def ch_requal(c) -> str:
    d = c.d
    rows = []
    for r in d.requal.itertuples():
        rows.append([f"<b>{r.category}</b>", f"{int(r.n_good_current_lot)}", f"{r.current_lot_median:.1f} vs {r.qualification_median:.1f}",
                     f"{r.p_value:.4f}", "<b>lot change</b>" if r.lot_shift else "no change",
                     "re-qualified model" if r.deployed else "qualification model",
                     f"{pct(r.base_good_check_rate)} → {pct(r.requal_good_check_rate)}",
                     f"{pct(r.base_escape_rate)} → {pct(r.requal_escape_rate)}"])
    lot = d.lots.get("carpet")
    lot_txt = (f"<p>For carpet, {lot['n_good']} good parts of the current lot added {lot['added_patches']:,} patch vectors "
               f"({100 * lot['coreset_ratio']:.0f}% coreset) to the bank; the re-fitted policy is threshold {lot['threshold']:g} ± "
               f"{lot['band']:g}. Shipped in <code>artifacts/carpet/requalified/</code> and used by default for PatchCore.</p>"
               if lot else "")
    body = f"""
<p>A new production lot — another carpet roll, a new batch of screws, a different supplier's glass — can look slightly
different while still being good. A good-only model then scores <i>good</i> parts higher and sends many of them to CHECK.
That is exactly what MVTec's test images show for carpet: they come from a different roll than the training images, and
the qualification model sent half of the good carpet parts to CHECK.</p>
{steps([("Collect", "A handful (~10) of <b>good</b> parts of the current lot — no defects needed. On MVTec the good parts of the "
                    "oracle split play this role; the operating split measures the effect."),
        ("Test", "A one-sided <b>Mann–Whitney U test</b> asks: do these parts score higher than the qualification's held-out good "
                 "parts? It compares ranks, so it needs no distribution assumption. p &lt; 0.01 → lot change."),
        ("Extend the memory", "A 10% greedy coreset of the new parts' patches joins the memory bank — the old lot stays normal "
                              "too."),
        ("New good reference", "Each new part is scored against the bank plus the <i>other</i> new parts (leave-one-out), so its "
                               "own patches do not make it look perfect."),
        ("Re-fit", "Synthetic defects are generated from the new parts; the score scale and the cost-optimal threshold are "
                   "re-fitted exactly as in chapter 11."),
        ("Release", "The lot model is stored with its date, source and test result; the app's <b>Production lot</b> switch selects "
                    "it, and the API uses it by default.")])}
{table(["product", "good parts of current lot", "median score: lot vs qualification", "p-value", "test", "deployed",
        "good parts sent to CHECK", "escapes"], rows, num_cols=(1, 3))}
{lot_txt}
{callout("key", "Honest trade-off", "<p>Re-qualification removes most of the carpet CHECK load (49.6% → 5.7% of all parts sent to "
         "CHECK), at the price of a few more escapes (5.3% → 8.8% on the operating split). Chapter 17 shows why that is the right trade on a real "
         "line: 12 inspectors per shift are not available, so the CHECK parts would not be looked at properly anyway.</p>")}
{callout("warn", "A rule fixed in advance", "<p>For screw the test found no lot change (p = 0.86), so the qualification model stays "
         "deployed — even though re-qualifying would also have lowered its escapes on the test split. Choosing the model by its "
         "test result would be leakage.</p>")}"""
    return chapter("requal", 12, "New production lots: re-qualification", body,
                   "When good parts of a new lot look different, update the reference of “good” — from good parts only.")


# ===================================================================================================== 13. drift
def ch_drift(c) -> str:
    d = c.d
    from defectspotter.drift import DRIFT_KINDS
    href = [[cat, *[f"{d.cal[cat]['health_reference'][s]['mean']:.1f} ± {d.cal[cat]['health_reference'][s]['std']:.1f}"
                    for s in ("brightness", "contrast", "sharpness")]] for cat in d.cats]
    sens = d.drift_sens[d.drift_sens.severity == 1.0]
    s_rows = []
    for kind, label in DRIFT_KINDS.items():
        row = [f"<b>{kind}</b><br><span class='small'>{e(label)}</span>"]
        for cat in d.cats:
            r = sens[(sens.category == cat) & (sens.drift == kind)]
            row.append("—" if r.empty else f"FR {pct(r.iloc[0].good_false_reject_rate, 0)} · flagged {pct(r.iloc[0].health_flag_rate, 0)}")
        s_rows.append(row)
    al = [[r.category, r.signal, f"{int(r.false_alarms_while_qualified)}",
           "—" if pd.isna(r.first_alarm_after_lamp_ageing) else f"part {int(r.first_alarm_after_lamp_ageing)}",
           "—" if pd.isna(r.first_alarm_after_camera_bump) else f"part {int(r.first_alarm_after_camera_bump)}"]
          for r in d.drift_alarms.itertuples()]
    body = f"""
<p>A model that learned “normal” calls anything unusual a defect — including a dimmer lamp, a defocused lens or a bumped
camera. Three mechanisms protect the line:</p>
{sub("drift-gate", "13.1", "Camera-health gate (every image)")}
<p>Brightness, contrast and sharpness of every image are compared with the qualified images:</p>
{formula("z = (value − mean_qualified) / std_qualified        any |z| > 4  →  no automatic decision: PASS/REJECT become CHECK")}
{table(["product", "brightness", "contrast", "sharpness (Laplacian variance)"], href,
       caption="Qualified reference (mean ± std of the model-training images, artifacts/&lt;product&gt;/calibration.json).")}
<p><b>Why CHECK and not REJECT?</b> On an image the model was not qualified for, a REJECT is most likely a lighting false
reject and a PASS could be an escape. A person decides, and the message says “check lamp / lens / camera”.</p>
{sub("drift-stress", "13.2", "Stress test: seven acquisition changes")}
<p>Every held-out good part was re-rendered with seven simulated changes at five severities. At full severity:</p>
{table(["change (severity 1.0)", *d.cats], s_rows,
       caption="FR = good parts the model alone would REJECT; flagged = share caught by the health gate (then sent to CHECK). "
               "Changes the gate misses (camera shift, colour temperature) are caught by the control charts below.")}
{sub("drift-spc", "13.3", "Control charts on the stream")}
{formula('''EWMA:   z_t = λ · x_t + (1 − λ) · z_(t−1)            λ = 0.2
limits: mean ± 3 · σ · √( λ/(2−λ) · (1 − (1−λ)^(2t)) )     alarm after 2 consecutive points outside
PSI = Σ (current_share − reference_share) · ln(current_share / reference_share)     10 quantile bins
      < 0.1 stable · 0.1–0.25 watch · > 0.25 act''')}
<p>A simulated 200-part stream per product: parts 1–60 qualified conditions, 61–120 a slowly ageing lamp, 121–150 after
maintenance, 151–200 a bumped camera.</p>
{table(["product", "signal", "false alarms while qualified", "first alarm after lamp ageing starts (part 61)",
        "first alarm after camera bump (part 151)"], al)}
<p class="small">The brightness charts of screw and carpet already alarm in the “qualified” phase: the stream uses test captures,
which are darker (screw) or brighter (carpet) than the training captures — the same capture difference that motivates
re-qualification (chapter 12). That is what such a chart is for.</p>
{callout("code", "In the code", "<p><code>defectspotter/drift.py</code> — <code>DRIFT_KINDS</code>, <code>apply_drift</code>, "
         "<code>health_check</code>, <code>EWMAChart</code>, <code>population_stability_index</code>, <code>first_alarm</code>.</p>")}"""
    return chapter("drift", 13, "Monitoring and drift", body,
                   "Keeping a good-only model honest after deployment: input validation, control charts, stability index.")


# ===================================================================================================== 14. few-shot
def ch_fewshot(c) -> str:
    d = c.d
    g = d.few_shot.groupby("shots").agg(auroc=("image_auroc", "mean"), lo=("image_auroc", "min"), hi=("image_auroc", "max"),
                                        pixel=("pixel_auroc", "mean"), bank=("bank_vectors", "mean"),
                                        frr=("false_reject_rate", "mean"), esc=("escape_rate", "mean"), n=("repeat", "count"))
    rows = [[f"{int(k)}", f"{r.auroc:.3f}", f"{r.lo:.3f}–{r.hi:.3f}", f"{r.pixel:.3f}", f"{r.bank:,.0f}",
             pct(r.frr) if pd.notna(r.frr) else "—", pct(r.esc) if pd.notna(r.esc) else "—", f"{int(r.n)}"] for k, r in g.iterrows()]
    body = f"""
<p>PatchCore needs no network training, so a new product line can be onboarded from a few good images in seconds: build the
bank from them, score each with the others (leave-one-out) to learn the good-score distribution, make synthetic defects
from them to set the threshold. The held-back <b>metal nut</b> is the test case:</p>
{table(["good images", "mean image AUROC", "range over draws", "pixel AUROC", "bank vectors", "false rejects", "escapes", "draws"],
       rows, num_cols=(0, 1, 3, 4, 5, 6, 7), caption="reports/few_shot.csv — random draws of the good images, evaluated on the metal-nut test set.")}
{callout("what", "What is happening", "<p>With 1 image the bank knows one pose and one lighting; with 8 it already covers most normal "
         f"variation (mean AUROC {g.loc[8, 'auroc']:.3f} at 8 shots). The curve flattens after 16 images. The app's <b>🆕 New product</b> "
         "tab runs this live: upload a few good images, press <i>Build model</i>, inspect.</p>" if 8 in g.index else "")}"""
    return chapter("fewshot", 14, "Few-shot onboarding of a new product", body,
                   "From zero to a working inspection in seconds — with 8 good images.")


# ===================================================================================================== 15. metrics
def ch_metrics(c) -> str:
    body = f"""
{table(["metric", "question it answers", "how it is computed", "good value"], [
        ["<b>Image AUROC</b>", "Are defective parts ranked above good ones?", "probability that a random defective part scores higher "
         "than a random good part = area under the ROC curve; threshold-free", "1.0 (0.5 = guessing)"],
        ["95% bootstrap interval", "How certain is the AUROC with ≈ 50–100 test images?", "resample the test set 1,000 times "
         "(stratified by class), 2.5th–97.5th percentile", "narrow"],
        ["<b>ROC curve</b>", "What does each threshold trade?", "true-positive rate (defects caught) vs false-positive rate (good flagged) "
         "as the threshold moves", "hugs the top-left"],
        ["Average precision (AP)", "Ranking quality with many good parts", "area under the precision–recall curve", "1.0"],
        ["<b>Pixel AUROC</b>", "Does the heat map light up on the defect?", "AUROC over pixels: ground-truth mask pixels vs normal pixels", "1.0"],
        ["<b>AUPRO @ 30%</b>", "Are small defects localised as well as large ones?", "per-region overlap of the thresholded map with each "
         "defect region, averaged up to 30% false-positive pixels, normalised", "1.0"],
        ["<b>False-reject rate</b>", "How many good parts are scrapped?", "good parts → REJECT / good parts", "0%"],
        ["<b>Escape rate</b>", "How many defects ship?", "defective parts → PASS / defective parts", "0%"],
        ["<b>CHECK load</b>", "How busy are the inspectors?", "share of all parts sent to CHECK, at 1% defects", "≤ 5%"],
        ["<b>Confusion matrix</b>", "All decisions at once", "rows = truth (good / defective), columns = PASS / CHECK / REJECT", "diagonal-ish"],
        ["Expected cost / 1,000 parts", "What does the operating point cost?", "cost model of chapter 11 at 1% defects", "low"],
        ["Latency", "Can it keep up with the line?", "end-to-end ms per part (median, p95) on a 4-thread CPU", "≪ cycle time"]])}
{callout("why", "Why both kinds", "<p>AUROC measures the <i>model</i> (how well it separates); the operating rates measure the <i>system</i> "
         "(model + threshold + band). A detector can have a high AUROC and still behave badly if its threshold is chosen poorly — "
         "chapter 10 shows the pixel VAE rejecting 93% of good parts at its label-free threshold.</p>")}"""
    return chapter("metrics", 15, "Metrics explained", body, "What each number in this guide and in the app means.")


# ===================================================================================================== 16. results
def ch_results(c) -> str:
    d = c.d
    from defectspotter.constants import METHOD_LABELS
    m = d.metrics
    order = ["patchcore", "ensemble", "fvae", "vae", "mean"]
    res_rows = []
    for cat in d.cats:
        for meth in order:
            r = m[(m.category == cat) & (m.method == meth)]
            if r.empty:
                continue
            r = r.iloc[0]
            res_rows.append([cat if meth == "patchcore" else "", METHOD_LABELS.get(meth, meth),
                             f"<b>{r.image_auroc:.3f}</b>" if meth == "patchcore" else f"{r.image_auroc:.3f}",
                             f"{r.image_auroc_ci_low:.3f}–{r.image_auroc_ci_high:.3f}", f"{r.image_ap:.3f}",
                             f"{r.pixel_auroc:.3f}", f"{r.aupro_30:.3f}", f"{int(r.test_images)} ({int(r.test_defects)})"])
    conf_rows = []
    for cat in d.cats:
        rt = c.deployed_rates[cat]
        ng, nd = int(rt["n_good"]), int(rt["n_defect"])
        gp = round(ng * (1 - rt["false_reject_rate"] - rt["good_check_rate"]))
        gc, gr = round(ng * rt["good_check_rate"]), round(ng * rt["false_reject_rate"])
        dp, dc = round(nd * rt["escape_rate"]), round(nd * rt["defect_check_rate"])
        conf_rows.append([f"<b>{cat}</b>", rt["model"], f"{gp} / {gc} / {gr}", f"{dp} / {dc} / {nd - dp - dc}",
                          pct(rt["false_reject_rate"]), pct(rt["escape_rate"]), pct(rt["check_rate"])])
    pt = d.per_type[(d.per_type.method == "patchcore") & (d.per_type.defect_type != "good")]
    pt_rows = [[r.category, r.defect_type.replace("_", " "), f"{int(r.images)}", pct(r.reject_rate, 0), pct(r.check_rate, 0),
                pct(r.pass_rate, 0), f"{r.mean_score:.1f}"] for r in pt.itertuples()]
    lat = d.latency.pivot(index="method", columns="category", values="latency_ms_median")
    lat_rows = [[METHOD_LABELS.get(k, k), *[f"{lat.loc[k, cat]:.0f}" for cat in d.cats]] for k in order if k in lat.index]
    sz = d.sizes.copy()
    sz_rows = [[f"<code>{e(r.file)}</code>", f"{r.size_mb:.2f}"] for r in sz.sort_values("size_mb", ascending=False).itertuples()]
    hist = [["v1 — one recipe for all products", "1.000", "0.935", "0.880", "first full build"],
            ["v2 — per-product recipe (label-free search)", "1.000", "0.983", "0.940", "raw descriptors, 288 px for screw"],
            ["v3 — pose-normalised screws (deployed)", *[f"{d.metric(k, 'patchcore'):.3f}" for k in ("bottle", "screw", "carpet")],
             "+ feature VAE, lot re-qualification"]]
    body = f"""
<p>Everything here is measured on MVTec test images that never trained a model or chose a threshold.</p>
{sub("res-ranking", "16.1", "Ranking and localisation, all detectors")}
{table(["product", "detector", "image AUROC", "95% interval", "AP", "pixel AUROC", "AUPRO 30%", "test images (defective)"],
       res_rows, num_cols=(2, 3, 4, 5, 6, 7))}
{sub("res-decisions", "16.2", "Decisions of the deployed system")}
{table(["product", "model", "good parts: PASS / CHECK / REJECT", "defective: PASS / CHECK / REJECT", "false rejects", "escapes",
        "CHECK share (1% defects)"], conf_rows, num_cols=(4, 5, 6),
       caption="Operating test split. Counts are a confusion matrix in one line: an ideal row is “all / 0 / 0” for good parts and "
               "“0 / 0 / all” for defective ones.")}
{fig(c.shot("confusion"), "<b>The same matrices in the app</b> (📈 Results → “Decisions of the deployed system”).", "confusion matrices", "shot")}
{sub("res-types", "16.3", "By defect type (PatchCore)")}
{table(["product", "defect type", "images", "REJECT", "CHECK", "PASS (escape)", "mean score"], pt_rows, num_cols=(2, 3, 4, 5, 6),
       caption="Qualification-lot models at the label-free threshold (reports/per_defect_type.csv).")}
{sub("res-qual", "16.4", "What the heat maps look like")}
{"".join(fig(c.figure(f"qualitative_{k}.jpg", 1300), f"<b>{k}</b>: test images, PatchCore heat maps and ground-truth masks",
             f"qualitative {k}") for k in d.cats)}
{sub("res-speed", "16.5", "Speed and footprint")}
<div class="grid2"><div>{table(["detector", *[f"{k} (ms)" for k in d.cats]], lat_rows, num_cols=(1, 2, 3),
                             caption="Median end-to-end latency per part, 4-thread CPU.")}</div>
<div>{details(f"All {len(sz_rows)} artifact files ({d.summary['artifact_total_mb']:.1f} MB)", table(["file", "MB"], sz_rows, num_cols=(1,)))}</div></div>
{sub("res-history", "16.6", "How the project got here")}
{table(["version", "bottle", "screw", "carpet", "what changed"], hist, num_cols=(1, 2, 3), hl_rows=(2,),
       caption="PatchCore image AUROC per version.")}"""
    return chapter("results", 16, "Results", body, "The numbers, on held-out real defects.")


# ===================================================================================================== 17. real world
def ch_realworld(c) -> str:
    from defectspotter.operations import SOURCES, shift_table
    sc = c.scenario
    tab = shift_table(c.rw_rows, sc)
    rows = [list(map(str, r)) for r in tab.values.tolist()]
    srcs = "".join(f"<li>{e(t)}" + (f" — <a href='{u}'>{e(u)}</a>" if u else "") + "</li>" for t, u in SOURCES.values())
    carpet_before = next((v for p, mdl, v in c.rw_views if p == "carpet" and "before" in mdl), None)
    carpet_after = next((v for p, mdl, v in c.rw_views if p == "carpet" and "re-qualified" in mdl), None)
    extra = ""
    if carpet_before and carpet_after:
        extra = callout("key", "Was lowering the human workload right? Yes.",
                        f"<p>Before re-qualification carpet would have needed about <b>{carpet_before['inspectors_needed']:.1f} "
                        f"inspectors</b> per shift; a station has 2–3, so most flagged parts would never be looked at properly "
                        f"(and an inspector who sees good parts all day stops looking carefully). Re-qualification brings it to "
                        f"<b>{carpet_after['inspectors_needed']:.1f}</b>. The honest price: about "
                        f"{carpet_after['defects_escaping'] - carpet_before['defects_escaping']:.0f} more defective carpet sections "
                        f"escape per shift ({carpet_after['defects_escaping']:.0f} instead of {carpet_before['defects_escaping']:.0f}) "
                        f"— still far better than 100% manual inspection ({c.manual['inspectors_needed']:.0f} inspectors, "
                        f"{c.manual['defects_escaping']:.0f} escapes). A plant that can staff more inspectors raises <i>Inspector "
                        f"capacity</i> in the threshold lab; the CHECK band widens and buys those escapes back.</p>")
    body = f"""
<p>Rates are easier to judge as counts. The deployed operating points are turned into <b>one 8-hour shift</b> of a realistic
inspection station (all values editable in the app's <b>Threshold &amp; cost</b> tab):</p>
{table(["assumption", "value", "source / reasoning"], [
        ["line speed", f"{sc.parts_per_hour:,.0f} parts/hour → {sc.parts_per_shift:,.0f} per shift", "a mid-speed station"],
        ["defect rate", f"{100 * sc.defect_rate:.0f}% → {sc.parts_per_shift * sc.defect_rate:,.0f} defective parts per shift", "typical first-pass yield"],
        ["human check", f"{sc.seconds_per_check:.0f} s → {sc.checks_per_inspector_hour:.0f} checks per inspector-hour", "pick the part, look with the heat map, decide"],
        ["inspector cost", f"{sc.inspector_cost_per_hour:.0f} USD/hour → <b>{sc.cost_per_check:.2f} USD per check</b>", "BLS median 22.82 USD/h (2024) + ~30% benefits and overhead"],
        ["inspector hit rate", f"{100 * sc.human_detection_rate:.0f}% of the defects they are shown", "industry average (See 2015)"],
        ["false reject", f"{sc.cost_false_reject:.0f} USD", "value of a scrapped mid-value part"],
        ["escape", f"{sc.cost_escape:.0f} USD", "1-10-100 rule of cost of quality: a complaint ≈ 1,000× the inspection cost"]])}
{table(list(tab.columns), rows, hl_rows=(len(rows) - 1,), caption="One shift at the deployed thresholds "
       "(defectspotter/operations.py — the same numbers as the app's Overview).")}
{extra}
<ul class="small">{srcs}</ul>
{fig(c.shot("overview_shift"), "<b>In the app</b>: Overview → “Real-world view: what one shift looks like”.", "real-world view", "shot")}"""
    return chapter("realworld", 17, "Real-world numbers: one production shift", body,
                   "Parts, people and money per shift — the language a plant manager speaks.")


# ===================================================================================================== 18. app tour
def ch_app(c) -> str:
    tour = [
        ("overview", "🏠 Overview", "The story on one page: the problem, how it works, eight key indicators (hover for definitions), "
         "the deployed operating table and the real-world shift view. The global settings bar above the tabs (Product, Detector, "
         "Production lot, Decision threshold, CHECK band, ↺ reset) applies to every page."),
        ("data", "📊 Data", "Where the images come from (MVTec AD, licence, pinned revision), the leakage-free split, defect types and "
         "sizes, acquisition statistics and the synthetic defects that replace real ones."),
        ("models", "🧠 Models & training", "What is trained and on what (the backbone is frozen, no head), the five detectors and how "
         "they differ, the backbone study, PatchCore's per-product recipe and search, VAE training curves, baseline and calibration."),
        ("recipe", "🧠 Models & training — recipe", "The deployed PatchCore recipe per product and what each design choice bought "
         "(best label-free objective per option)."),
        ("results", "📈 Results", "Model comparison (which detector is best), metric definitions, confusion matrices, AUROC bars with "
         "bootstrap intervals, ROC curves, heat-map quality, per-defect-type rates, accuracy vs speed."),
        ("threshold", "🎚️ Threshold & cost", "The trade-off priced in money: set the defect rate, costs, line speed, inspector hit rate "
         "and capacity; the tiles, distributions, rate curves and cost curve update live. “Apply cost-optimal threshold” uses no "
         "real defects."),
        ("drift", "🌗 Drift & new lots — A/B", "A: apply a lamp, focus, camera or colour change to a good part and see the model and the "
         "health gate react. B: a 200-part stream with control charts and PSI."),
        ("requalify", "🌗 Drift & new lots — C", "C: lot-shift test and re-qualification from good parts of the current lot — run it "
         "live with the sample parts."),
        ("new_product", "🆕 New product", "Few-shot onboarding of the metal nut: the learning curve and a live build from 8 images."),
        ("inspect", "🎬 Live demo — Inspect a part", "Upload, paste or photograph a part (or click a sample): decision card with the "
         "reason, score bar against the zones, heat map with numbered regions and the input with the ground-truth outline."),
        ("inspect_detail", "🎬 Live demo — details", "All five detectors on the same part, the pixel VAE's reconstruction, the pixel "
         "anomaly map, the suspected-region table and the camera-health check."),
        ("batch", "🎬 Live demo — Batch", "A whole tray at once: decision tiles, scores against the zones, a summary table, overlays and "
         "a CSV for the MES; CHECK parts go to the review queue automatically."),
        ("review", "🎬 Live demo — Review queue", "Human in the loop: pending CHECK parts, confirm or release with a note, audit log "
         "(CSV)."),
        ("api", "🎬 Live demo — API", "The three public endpoints with a Python example, the /health record and the artifact manifest."),
    ]
    items = "".join(fig(c.shot(name), f"<b>{title}.</b> {text}", title, "shot") for name, title, text in tour)
    body = f"""
<p>The app is organised as a presentation: from the story (left) to the live demonstration (last tab). Every page starts
with a box saying <i>what you see</i> and <i>how to read it</i>; every chart has a caption.</p>
{items}
{callout("tip", "Stale page after an update", "<p>If the app was redeployed while your page was open, a yellow banner asks you to "
         "reload. Reloading avoids the “didn't receive enough input values” errors a stale page would otherwise produce.</p>")}"""
    return chapter("app", 18, "The app, tab by tab", body, "What every page shows and how to use it.")


# ===================================================================================================== 19. demo
def ch_demo(c) -> str:
    d = c.d
    rows = []
    for r in d.demo.itertuples():
        f = str(r.file)
        p = c.root / "demo_images" / f
        thumb = f'<img src="{c.thumb(p, 96)}" alt="{e(f)}" style="width:56px;height:56px;border-radius:6px">' if p.is_file() else ""
        decision = str(r._4)
        dec_html = c.dec(decision) if decision in ("PASS", "CHECK", "REJECT") else e(decision)
        rows.append([thumb, f"<code>{e(f)}</code>", e(r.product), e(r._3), dec_html, e(r._5), e(r._7)])
    body = f"""
{sub("demo-3", "19.1", "Three-minute script")}
{steps([("Overview", "The problem (escapes vs false rejects), “learns only from good parts”, the eight indicators, the one-shift view."),
        ("Inspect a defect", "🎬 Live demo → Inspect a part, product <b>Bottle</b>: upload <code>demo_images/bottle/defect_broken_large_004.png</code> "
                             "→ REJECT with the heat map on the break. Point at the score bar and the “Why?” text."),
        ("Inspect a good part", "Upload <code>demo_images/bottle/good_017.png</code> → PASS."),
        ("Hard case", "Product <b>Screw</b>: upload <code>defect_scratch_head_009.png</code> → the scratch is localised although screws lie "
                      "at random angles (pose normalisation). Show “All detectors on this part”."),
        ("Threshold", "Move the threshold slider: the decision changes; ↺ restores the calibrated policy.")])}
{sub("demo-10", "19.2", "Ten-minute script (adds operations)")}
{steps([("Batch", "📦 Batch → Load sample batch → Inspect batch: tiles, score strip, CSV. CHECK parts land in the review queue."),
        ("Review", "🧑‍🔧 Review queue: pick a pending part, write a note, confirm or release → audit log."),
        ("Lamp failure", "Upload <code>demo_images/drift_demo/bottle_good_part_lamp_failure.png</code>: a good part under a failing lamp. "
                         "The camera-health gate refuses an automatic decision → CHECK."),
        ("New lot", "🌗 Drift &amp; new lots → C: <i>Load the sample current-lot good parts</i> → <i>Re-qualify</i> (carpet). Good carpet "
                    "parts that went to CHECK now PASS."),
        ("Which model?", "📈 Results → Model comparison: five detectors, PatchCore best; the feature VAE is the best generative model."),
        ("Money", "🎚️ Threshold &amp; cost: change the escape cost or the line speed and watch the cost-optimal threshold move.")])}
{sub("demo-images", "19.3", "Images to upload (demo_images/)")}
{table(["", "file", "product", "ground truth", "expected", "score", "what to show"], rows, num_cols=(5,),
       caption="All expected outcomes were verified against the deployed models (demo_images/expected_outcomes.csv). "
               "<code>carpet_new_lot_good_parts/</code> holds the re-qualification inputs.")}"""
    return chapter("demo", 19, "Live demonstration", body, "What to click and upload, in which order, and what will happen.")


# ===================================================================================================== 20. API
def ch_api(c) -> str:
    rec = json.dumps(c.api_example, indent=2, ensure_ascii=False)
    body = f"""
{table(["endpoint", "inputs", "returns"], [
        ["<code>/inspect_part</code>", "image, category (bottle · screw · carpet), method (patchcore · fvae · vae · mean · ensemble), "
         "threshold, band (empty = calibrated)", "JSON decision record + heat-map overlay image"],
        ["<code>/inspect_batch</code>", "list of image files, category, method", "per-part JSON rows + CSV file"],
        ["<code>/health</code>", "—", "version, categories, detectors, calibrated policies, which products are re-qualified"]])}
{code('''pip install gradio_client
from gradio_client import Client, handle_file
client = Client("bhkr0901/defectspotter")          # or "http://127.0.0.1:7860" for a local app
record, overlay = client.predict(handle_file("part.png"), "screw", "patchcore", None, None,
                                 api_name="/inspect_part")
print(record["decision"], record["score"], record["regions"])''', "python")}
<p>A real response (screw, <code>demo_images/screw/defect_scratch_head_009.png</code>, PatchCore):</p>
{code(rec, "json")}
<p>PatchCore requests use the current-lot model wherever the product was re-qualified (the same default as the app). Field
meanings: <code>score</code> on the 0–100 scale; <code>threshold</code>/<code>band</code> the policy used; <code>regions</code> the
suspected areas (pixel score &gt; 50, image coordinates of the 512-px display, area %, peak and mean pixel score);
<code>health</code> the camera-health z-scores; <code>detectors</code> the other detectors' opinions.</p>"""
    return chapter("api", 20, "Using it from code (API)", body, "Three public endpoints for MES integration, scripts and tests.")


# ===================================================================================================== 21. run
def ch_run(c) -> str:
    body = f"""
{sub("run-live", "21.1", "Use the live app")}
<p>Open <a href="https://huggingface.co/spaces/bhkr0901/defectspotter">huggingface.co/spaces/bhkr0901/defectspotter</a> (direct:
<a href="https://bhkr0901-defectspotter.hf.space/">bhkr0901-defectspotter.hf.space</a>). Nothing to install.</p>
{sub("run-local", "21.2", "Run it on your computer (CPU only)")}
{code('''git clone https://github.com/hbasired/defectspotter.git
cd defectspotter
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt                        # NumPy, OpenCV, ONNX Runtime, Gradio, Plotly
python app.py                                          # http://127.0.0.1:7860''', "bash")}
<p>The trained models (<code>artifacts/</code>), the evidence the app plots (<code>reports/</code>) and sample images
(<code>examples/</code>) are in the repository, so the app works immediately — no GPU, no PyTorch.</p>
{sub("run-train", "21.3", "Re-train everything")}
{code('''pip install -r requirements-train.txt                  # + PyTorch, torchvision, ONNX export
python scripts/download_data.py --dest data/mvtec_ad   # MVTec AD from the pinned Hugging Face mirror (1.3 GB)
python scripts/train.py --data data/mvtec_ad --stages index eda synthetic
pip install -r requirements-diffusion.txt              # optional: Stable Diffusion defects
python scripts/generate_diffusion_defects.py --data data/mvtec_ad --per-category 12
python scripts/train.py --data data/mvtec_ad --vae-epochs 120 \\
       --stages patchcore vae fvae mean severity diffusion calibrate evaluate requalify latency drift few_shot examples summary
python scripts/summarize_results.py --write            # refresh result tables in README / handoff
python scripts/build_report.py                         # static HTML report
python scripts/make_demo_images.py                     # demo_images/ for presentations
python scripts/build_guide.py                          # this guide
pytest                                                 # tests''', "bash")}
<p>CPU ≈ 2–2.5 h on 4 cores (a GPU is ≈ 5× faster). Every stage caches its inputs in <code>../defectspotter_work</code>; the long
stages are resumable per product, so a single stage can be re-run after a change.</p>
{sub("run-colab", "21.4", "Google Colab")}
<p>Open <code>DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb</code> and run the cells top to bottom: it clones the
repository, runs every stage with inline charts (including the five-detector comparison and the one-shift view), launches
the app with a public link and can deploy to Hugging Face.</p>
{sub("run-deploy", "21.5", "Deploy to Hugging Face")}
{code('''HF_TOKEN=<write token> python scripts/deploy_hf.py --space <user>/defectspotter''', "bash")}
<p>The script stages <code>app.py</code>, the <code>defectspotter</code> package, <code>ui</code>, <code>artifacts</code>,
<code>reports</code>, <code>examples</code> and <code>docs</code> with a Space card, and uploads them. Details:
<code>DefectSpotter_HuggingFace_Deployment_Guide.md</code>.</p>"""
    return chapter("run", 21, "Run, re-train, deploy", body, "Commands for every way of running the project.")


# ===================================================================================================== 22. repo
def ch_repo(c) -> str:
    modules = [
        ["defectspotter/data.py", "indexing MVTec, leakage-free splits, image I/O, acquisition statistics"],
        ["defectspotter/features.py", "ResNet-18 feature extraction, multi-output ONNX export"],
        ["defectspotter/patchcore.py", "memory bank, greedy coreset (CoresetLadder), nearest neighbour, scoring"],
        ["defectspotter/pose.py", "pose normalisation of elongated parts and the inverse warp of maps"],
        ["defectspotter/vae.py · fvae.py", "pixel VAE and feature-space VAE: models, training, ONNX export"],
        ["defectspotter/baseline.py", "ECC-aligned mean / std baseline"],
        ["defectspotter/synthetic.py", "foreground mask, cut-paste / scratch / stain defects"],
        ["defectspotter/decision.py", "score and pixel scales, decisions, cost model, KDE smoothing, threshold policies"],
        ["defectspotter/drift.py", "acquisition changes, health gate, EWMA, PSI"],
        ["defectspotter/operations.py", "one-shift real-world view (parts, people, money)"],
        ["defectspotter/metrics.py", "AUROC with bootstrap, AP, pixel AUROC, AUPRO, ROC points"],
        ["defectspotter/pipeline.py", "the 17 training / evaluation stages"],
        ["defectspotter/runtime.py", "Inspector: inference, batch, lot re-qualification, few-shot onboarding"],
        ["ui/app.py · explain.py · charts.py · story.py · theme.py · store.py", "the Gradio app: layout, explanations, charts, theme, report access"],
        ["scripts/", "train, download, deploy, diffusion defects, demo images, report, guide, backbone study, summaries"],
        ["tests/", "core unit tests and app tests (run in GitHub Actions)"]]
    tree = """defectspotter/
├── app.py                         entry point (local, Colab, Hugging Face)
├── defectspotter/                 the library (training + inference)
├── ui/                            the Gradio app
├── scripts/                       command-line tools (train, deploy, build_guide …)
├── artifacts/                     trained models: feature_extractor.onnx + per product bank, VAEs, baseline, calibration
│   └── carpet/requalified/        the current-lot model
├── reports/                       every number and figure the app and this guide show (CSV, JSON, figures/)
├── examples/                      sample images for the app (test parts, synthetic, diffusion, re-qualification, new product)
├── demo_images/                   images to upload in a live demo + expected outcomes
├── docs/                          GUIDE.md, this HTML guide, model card, data card, screenshots
├── tests/                         pytest suite
├── DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb
├── DefectSpotter_HuggingFace_Deployment_Guide.md
└── PROJECT_HANDOFF.md             status, decisions and next steps for the next engineer"""
    body = f"""
<pre class="tree"><code>{e(tree)}</code></pre>
{table(["module", "role"], [[f"<code>{e(a)}</code>", b] for a, b in modules])}"""
    return chapter("repo", 22, "Repository map", body, "Where everything lives.")


# ===================================================================================================== 23. lessons
def ch_lessons(c) -> str:
    rows = [
        ["Bottle picked an expensive recipe on noise", "objective differences of 0.001 decided between settings", "parsimony rule: cheapest "
         "search within 0.005 of the best", "same accuracy, faster and smaller"],
        ["Synthetic defects on the empty background", "percentile foreground mask leaked to the tray", "Otsu threshold on the colour distance "
         "to the background", "defects always on the part"],
        ["Unstable thresholds", "≈ 50 good parts → false-reject rate in 2% steps", "kernel-smoothed rates (Silverman bandwidth)",
         "stable, reproducible thresholds"],
        ["50% of parts to a human (carpet)", "the test captures come from another roll (lot change)", "planning share 0.5 of capacity + "
         "lot re-qualification from good parts", "CHECK 49.6% → 5.7% of good carpet parts"],
        ["Screw escapes (≈ 9% in v2)", "random orientation; the bank had to cover every angle", "pose normalisation (PCA axis, head-left)",
         "screw AUROC 0.983 → 0.997"],
        ["Long screw tips cut off after rotation", "centring on the centroid (near the heavy head)", "centre the part's extent; pixel-centre "
         "warp convention", "screw pixel AUROC 0.996"],
        ["Feature VAE error exploding", "near-constant feature channels (std ≈ 0) divided by ≈ 0", "std floor at 10% of the median channel std",
         "stable maps; verified on real features"],
        ["Hours of coreset computation", "one greedy run per ratio and per candidate set", "CoresetLadder: one run serves all ratios; "
         "300 k candidate pool", "recipe search feasible on a CPU"],
        ["Container restarts killed long runs", "2-hour limits and restarts in the build environment", "resumable stages with done-markers "
         "and per-product caches", "a restart costs minutes, not hours"],
        ["Errors after a redeploy", "a page opened before the update talks to a server with new event wiring", "build fingerprint + "
         "“reload” banner", "no confusing errors after updates"]]
    body = table(["problem", "cause", "fix", "effect"], rows)
    return chapter("lessons", 23, "Engineering decisions and lessons learned", body,
                   "The problems met on the way and how each one was solved — useful when presenting “why it is built like this”.")


# ===================================================================================================== 24. responsible
def ch_responsible(c) -> str:
    body = """
<ul><li><b>Human in the loop.</b> Uncertain parts go to a person; every verdict is logged. The model is never silently
retrained from verdicts.</li>
<li><b>No automatic decision on unqualified images.</b> The camera-health gate turns PASS / REJECT into CHECK when the
acquisition is out of range.</li>
<li><b>Thresholds chosen without real defects</b>, measured on images that played no role in any choice.</li>
<li><b>The score is a ranking, not a probability.</b> A score of 80 does not mean “80% defective”.</li>
<li><b>Heat maps show visual deviation, not root cause.</b> A highlighted area is unusual; why it is unusual is for the
inspector and the process engineer.</li>
<li><b>Synthetic defects do not cover every real failure mode.</b> The detection-limit replays and the capability gate make
this visible instead of hiding it.</li>
<li><b>Small test sets.</b> 54–105 operating images per product — rates move in steps of 1–2%; AUROCs carry bootstrap intervals.</li>
<li><b>What the test set was used for.</b> Design options (e.g. which backbones to compare) were proposed after looking at test
results; the choice <i>within</i> the options is label-free. The backbone study is reported as analysis, not as selection evidence.</li>
<li><b>Licence.</b> MVTec AD is CC BY-NC-SA 4.0: research and teaching only. A production system needs the plant's own images.</li></ul>"""
    return chapter("responsible", 24, "Responsible use and limitations", body, "What the system does not do, and the safeguards around it.")


# ===================================================================================================== 25. FAQ
def ch_faq(c) -> str:
    qa = [
        ("Is the backbone fine-tuned?", "No. ResNet-18 keeps its ImageNet weights; no head is attached."),
        ("So what is trained?", "The memory bank is <i>built</i> from good parts (no gradients); the two VAEs are trained from scratch on "
         "good parts; the baseline computes statistics; the calibration fits a score scale and thresholds."),
        ("Did we build all five detectors?", "Yes. All five are separate, calibrated models that can be selected in the app and the API; "
         "PatchCore is deployed because it is the best (chapter 10)."),
        ("Does the ResNet feed the VAE?", "Only the feature-space VAE: it learns to rebuild the ResNet's feature map (image → ResNet → "
         "VAE). The pixel VAE rebuilds the raw image and never sees the ResNet; the mean baseline doesn't either. PatchCore and the "
         "feature-space VAE share the same frozen ResNet but are separate detectors."),
        ("Is the ResNet compared with ImageNet?", "No. ImageNet is the photo collection the ResNet weights were learned on (by the "
         "torchvision team, before this project). The backbone study compares ResNet-18 with three other ImageNet-pre-trained "
         "networks to choose the feature extractor; none of them is trained here."),
        ("Were real defects used anywhere?", "Only for evaluation and for the comparison “oracle” threshold — never to train a model or "
         "set the deployed threshold."),
        ("Why PASS / CHECK / REJECT and not good / bad?", "Uncertain parts are cheaper to check than to scrap or ship; the band absorbs "
         "the overlap of the score distributions."),
        ("Why is the threshold not 50?", "50 is just the middle of the scale. The cost model puts the threshold where the expected cost is "
         "lowest — usually in the high 30s, because an escape costs 200× a false reject."),
        ("What happens with a new lot or a camera change?", "The health gate and the control charts raise an alarm; a camera problem is "
         "fixed at the camera, a new lot is re-qualified from ~10 good parts (chapters 12–13)."),
        ("Can it name the defect type?", "No — it finds the unusual area. Naming types would need labelled defects (a classifier on top "
         "of confirmed CHECK/REJECT parts is a natural next step)."),
        ("Why ONNX?", "One portable file per model, run by the light ONNX Runtime: no PyTorch, no GPU, identical results on a laptop, a "
         "Space or an industrial PC."),
        ("How fast is it?", "84–136 ms per part with PatchCore on a 4-thread CPU, i.e. 440–710 parts per minute."),
        ("How big is it?", "≈ 57 MB in total; the shared backbone is 10.6 MB and a product's bank 1–2.4 MB."),
        ("Why did I see errors after switching the detector?", "The app had been updated while the page was open. Reload the page; the app "
         "now shows a reload banner in that case."),
        ("Can I add my own product?", "Yes: 🆕 New product builds a model from a handful of good images; for production, run the "
         "pipeline on your own image folders."),
    ]
    body = "".join(details(e(q), f"<p>{a}</p>", open_=i < 3) for i, (q, a) in enumerate(qa))
    return chapter("faq", 25, "Frequently asked questions", body)


# ===================================================================================================== 26. glossary
GLOSSARY = [
    ("Anomaly detection", "Learning what normal looks like and flagging deviations.", "problem"),
    ("AP (average precision)", "Area under the precision–recall curve.", "metrics"),
    ("AUPRO", "Per-region overlap of the heat map with each defect, up to 30% false-positive pixels.", "metrics"),
    ("AUROC", "Probability that a random defective part scores above a random good part.", "metrics"),
    ("Backbone", "The pre-trained network that turns an image into features (here ResNet-18, frozen).", "backbone"),
    ("Band (CHECK band)", "Half-width b around the threshold inside which parts go to a human.", "decision"),
    ("Bootstrap interval", "Uncertainty range from re-sampling the test set.", "metrics"),
    ("Calibration", "Fitting the score scale and thresholds on held-out good parts + synthetic defects.", "decision"),
    ("Capability gate", "A defect set influences the threshold only if its AUROC ≥ 0.9.", "synthetic"),
    ("CNN", "Convolutional neural network: stacked learned filters.", "backbone"),
    ("Coreset (greedy k-center)", "A small subset that covers a large set: repeatedly add the least-covered point.", "patchcore"),
    ("Cost model", "Expected cost per part from false rejects, escapes and checks at a defect rate.", "decision"),
    ("CutPaste", "Synthetic defect: a patch of the part pasted elsewhere.", "synthetic"),
    ("Diffusion model", "Generates images by removing noise step by step (Stable Diffusion).", "synthetic"),
    ("ECC", "Enhanced correlation coefficient image alignment (baseline).", "baseline"),
    ("Ensemble", "Weighted mix of PatchCore and feature-VAE scores.", "compare"),
    ("Escape", "A defective part that is passed.", "problem"),
    ("EWMA", "Exponentially weighted moving average control chart.", "drift"),
    ("False reject (FRR)", "A good part that is rejected; FRR = rate.", "problem"),
    ("Feature map / descriptor", "The vector a CNN layer computes for each image patch (here 384 numbers).", "backbone"),
    ("Few-shot", "Building a model from a handful of examples.", "fewshot"),
    ("Frozen", "Weights not changed during this project.", "backbone"),
    ("Health gate", "Brightness / contrast / sharpness z-score check of every image.", "drift"),
    ("Heat map (anomaly map)", "Per-pixel anomaly score drawn over the image.", "patchcore"),
    ("ImageNet", "1.28 million labelled everyday photos used to pre-train the backbone.", "backbone"),
    ("Johnson–Lindenstrauss projection", "Random projection to fewer dimensions that roughly preserves distances.", "patchcore"),
    ("KDE", "Kernel density estimate: a smooth estimate of a score distribution.", "decision"),
    ("KL divergence", "How different two probability distributions are; the VAE's regulariser.", "vae"),
    ("Latent", "The compressed code inside an autoencoder.", "vae"),
    ("LCM-LoRA", "Small adapter that lets Stable Diffusion work in ~6 steps.", "synthetic"),
    ("Leave-one-out", "Scoring each item with a model built from all the others.", "requal"),
    ("Lot", "A batch of production (roll, batch, supplier) that may look slightly different.", "requal"),
    ("Mann–Whitney U", "Rank test of whether one sample tends to be larger than another.", "requal"),
    ("Memory bank", "Stored patch descriptors of good parts (PatchCore).", "patchcore"),
    ("MVTec AD", "Industrial anomaly detection benchmark used here.", "data"),
    ("Nearest neighbour", "The closest stored vector; its distance is the patch anomaly.", "patchcore"),
    ("ONNX", "Portable model format run by ONNX Runtime.", "backbone"),
    ("Oracle threshold", "The threshold real defect labels would give — for comparison only.", "decision"),
    ("PatchCore", "Memory-bank anomaly detector (Roth et al. 2022) — the deployed detector.", "patchcore"),
    ("Pose normalisation", "Rotating each screw to a canonical orientation before feature extraction.", "patchcore"),
    ("PSI", "Population stability index of the score distribution.", "drift"),
    ("Re-qualification", "Updating the reference of “good” for a new lot from good parts.", "requal"),
    ("Reparameterisation trick", "z = μ + σ·ε, so sampling stays differentiable.", "vae"),
    ("ResNet", "CNN with residual (shortcut) connections.", "backbone"),
    ("SDEdit", "Editing an image with a diffusion model by partial noising and denoising.", "synthetic"),
    ("Score scale (0–100)", "25 = typical good part, 75 = typical synthetic defect.", "decision"),
    ("SSIM", "Structural similarity index (brightness, contrast, structure).", "vae"),
    ("Synthetic defect", "A painted defect with an exact mask, used instead of real defects.", "synthetic"),
    ("VAE", "Variational autoencoder: an autoencoder with a random, regularised latent.", "vae"),
]


def ch_glossary(c) -> str:
    items = "".join(f'<dt>{e(t)}</dt><dd>{e(dsc)} <a href="#{a}">→ ch.</a></dd>' for t, dsc, a in GLOSSARY)
    return chapter("glossary", 26, "Glossary", f'<dl class="gloss">{items}</dl>')


# ===================================================================================================== 27. references
def ch_refs(c) -> str:
    refs = [
        "Bergmann, Batzner, Fauser, Sattlegger, Steger — The MVTec Anomaly Detection Dataset. IJCV 129 (2021); CVPR 2019.",
        "Roth, Pemula, Zepeda, Schölkopf, Brox, Gehler — Towards Total Recall in Industrial Anomaly Detection (PatchCore). CVPR 2022.",
        "He, Zhang, Ren, Sun — Deep Residual Learning for Image Recognition. CVPR 2016.",
        "Kingma, Welling — Auto-Encoding Variational Bayes. ICLR 2014.",
        "Li, Sohn, Yoon, Pfister — CutPaste: Self-Supervised Learning for Anomaly Detection and Localization. CVPR 2021.",
        "Rombach et al. — High-Resolution Image Synthesis with Latent Diffusion Models (Stable Diffusion). CVPR 2022.",
        "Meng et al. — SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations. ICLR 2022.",
        "Luo et al. — LCM-LoRA: A Universal Stable-Diffusion Acceleration Module. 2023.",
        "Sener, Savarese — Active Learning for CNNs: A Core-Set Approach (greedy k-center). ICLR 2018.",
        "Evangelidis, Psarakis — Parametric Image Alignment Using Enhanced Correlation Coefficient Maximization. TPAMI 2008.",
        "Wang, Bovik, Sheikh, Simoncelli — Image Quality Assessment: From Error Visibility to Structural Similarity (SSIM). TIP 2004.",
        "Silverman — Density Estimation for Statistics and Data Analysis. 1986.",
        "Mann, Whitney — On a Test of Whether One of Two Random Variables is Stochastically Larger than the Other. 1947.",
        "Roberts — Control Chart Tests Based on Geometric Moving Averages (EWMA). Technometrics 1959.",
        "See — Visual Inspection Reliability for Precision Manufactured Parts. Human Factors 57 (2015).",
        "U.S. Bureau of Labor Statistics — Occupational Outlook Handbook: Quality Control Inspectors (median pay 2024).",
        "Labovitz, Chang, Rosansky — Making Quality Work (1-10-100 rule). 1992.",
    ]
    return chapter("refs", 27, "References and sources", "<ol>" + "".join(f"<li>{e(r)}</li>" for r in refs) + "</ol>")


CHAPTERS = [("Part III · Decisions and operations", [ch_decision, ch_requal, ch_drift, ch_fewshot]),
            ("Part IV · Results", [ch_metrics, ch_results, ch_realworld]),
            ("Part V · Use it", [ch_app, ch_demo, ch_api, ch_run, ch_repo]),
            ("Part VI · Reference", [ch_lessons, ch_responsible, ch_faq, ch_glossary, ch_refs])]
