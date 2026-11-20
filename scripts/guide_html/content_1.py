"""Guide chapters 1-10: the project, the flows, the data and every model."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import svg
from .helpers import (callout, chapter, code, details, e, f3, fig, formula, kpis, part, pct, steps, sub, table, thumbs,
                      zones)


# ===================================================================================================== 1. one page
def ch_overview(c) -> str:
    d, n = c.d, c.n
    pc = {k: d.metric(k, "patchcore") for k in d.cats}
    dep = {k: c.deployed_rates[k] for k in d.cats}
    lat = d.latency[d.latency.method == "patchcore"].set_index("category")
    tiles = kpis([
        (f"{d.macro('patchcore'):.3f}", "Detection quality (image AUROC)",
         " · ".join(f"{k} {pc[k]:.3f}" for k in d.cats) + " — 1.0 = every defect ranked above every good part"),
        (f"{d.macro('patchcore', 'pixel_auroc'):.3f}", "Localisation (pixel AUROC)", "does the heat map light up on the defect?"),
        ("0%", "Good parts scrapped", "false rejects at the deployed thresholds, all products"),
        (" / ".join(pct(dep[k]["escape_rate"], 1) for k in d.cats), "Defects missed (escapes)",
         " / ".join(d.cats) + " — share of defective parts that PASS"),
        (" / ".join(pct(dep[k]["check_rate"], 1) for k in d.cats), "Sent to a human (CHECK)",
         "share of all parts at 1% defects; carpet after re-qualification"),
        (f"{lat.latency_ms_median.min():.0f}–{lat.latency_ms_median.max():.0f} ms", "Per part on a CPU",
         "median, 4 threads, whole pipeline incl. heat map"),
        (f"{n['artifact_mb']:.0f} MB", "All models together", f"backbone {d.manifest['feature_extractor']['size_mb']:.1f} MB + "
                                                               "banks, VAEs, baselines for 3 products"),
        (f"{c.saving_vs_manual:.0f}%", "Cheaper than 100% manual", "cost per shift vs an inspector looking at every part "
                                                                     "(chapter 17)"),
    ])
    body = f"""
{tiles}
<p><b>DefectSpotter</b> is a visual inspection system for the end of a production line. It learns what <b>good</b> parts
look like — from good parts only — and flags anything unusual. For every part it returns an <b>anomaly score</b> on an
easy 0–100 scale, a <b>heat map</b> showing where the part looks unusual, and a three-way decision:
{c.dec('PASS')} ship it, {c.dec('CHECK')} let a human inspector decide, {c.dec('REJECT')} quarantine it.
Not a single defect image is needed — neither to train the models nor to choose the decision thresholds.</p>
{svg.figure(svg.system_overview(n), "<b>The whole system in one picture.</b> Lane 1 runs once per product (offline); lane 2 "
            "runs for every part in milliseconds; lane 3 keeps the system honest after deployment. Every box is explained "
            "in chapter 3 (flows) and chapters 4–14 (details).", ("data", "model", "proc", "gate", "human", "store"))}
{svg.figure(svg.detector_map(n), "<b>Which detector uses what.</b> Five separate paths, not one chain: the frozen ResNet-18 feeds "
            "only PatchCore and the feature-space VAE; the pixel VAE and the mean-image baseline read raw pixels; the ensemble "
            "mixes the scores of ① and ②. The <b>Detector</b> menu in the app chooses which path decides.",
            ("data", "model", "best", "proc", "gate"))}
{callout("why", "ResNet, VAE and ImageNet — how they relate",
         "<p><b>Is the ResNet trained?</b> No. Its weights were learned on <b>ImageNet</b> (1.28 million everyday photos) by the "
         "torchvision team, long before this project; here it is used <b>frozen</b>, as a fixed feature extractor. "
         "<b>Is it compared with ImageNet?</b> No — ImageNet is a photo collection, not a model. The <i>backbone study</i> "
         "(chapter 6.3) compares ResNet-18 with three other ImageNet-pre-trained networks to choose the feature extractor. "
         "<b>Does the ResNet feed the VAE?</b> Only the <b>feature-space VAE</b> (it learns to rebuild the ResNet's feature "
         "map). The <b>pixel VAE</b> learns to rebuild the raw image and never sees the ResNet. What is actually trained here: "
         "the two VAEs (from scratch, good parts only); the PatchCore memory bank is <i>built</i>; the baseline computes "
         "statistics.</p>")}
<div class="cols">
 <div class="card"><h4>What was built</h4><ul>
  <li>Five anomaly detectors, all trained or built from good parts and calibrated the same way: <b>PatchCore</b>
  (deployed), a <b>feature-space VAE</b> and a <b>pixel VAE</b> (generative AI), a <b>mean-image baseline</b> and an
  <b>ensemble</b> — compared in chapter 10.</li>
  <li>A label-free <b>calibration</b>: 0–100 score scale and cost-optimal PASS / CHECK / REJECT thresholds from held-out
  good parts and <b>synthetic defects</b> (procedural + Stable Diffusion).</li>
  <li><b>Operations</b>: camera-health gate, control charts, human review queue with audit log, lot re-qualification,
  few-shot onboarding of a new product.</li>
  <li>A <b>Gradio app</b> (9 tabs, live demo last), a public <b>API</b>, a <b>Colab notebook</b>, a <b>Hugging Face
  Space</b>, tests and CI.</li></ul></div>
 <div class="card"><h4>How to read this guide</h4><ul>
  <li><b>Presenting it?</b> Chapters 1, 2, 10, 17, then 18–19 (app tour and demo script).</li>
  <li><b>Want the mechanics?</b> Chapter 3 shows every flow step by step; chapters 6–13 explain each model and concept
  with its exact parameters.</li>
  <li><b>Running it?</b> Chapters 20–22.</li>
  <li>Boxes labelled <b>What is happening</b> narrate a step; <b>Why</b> boxes justify a choice; <b>In the code</b>
  points to the file.</li>
  <li>Use the search box (top left) to find a term — e.g. “KL”, “coreset”, “EWMA”.</li></ul></div>
</div>"""
    return chapter("overview", 1, "The project on one page", body,
                   "A camera looks at every part. DefectSpotter answers three questions in about a tenth of a second: "
                   "<i>is this part unusual, where, and what should happen to it?</i>")


# ===================================================================================================== 2. problem
def ch_problem(c) -> str:
    sc, manual = c.scenario, c.manual
    body = f"""
<p>A quality inspector at the end of a line must stop <b>defective</b> parts without throwing away <b>good</b> ones. Two
errors cost money in opposite directions:</p>
{table(["error", "what happens", "typical cost", "how DefectSpotter limits it"], [
        ["<b>Escape</b> (missed defect)", "a defective part passes and reaches the customer — complaint, return, warranty, recall",
         f"high: {sc.cost_escape:.0f} USD in this guide's scenario", "a cost model that weighs escapes 200× more than a false reject"],
        ["<b>False reject</b>", "a good part is scrapped or reworked", f"low per part: {sc.cost_false_reject:.0f} USD",
         "a CHECK band: uncertain parts go to a human instead of the scrap bin"],
        ["<b>Human check</b>", "an inspector looks at a flagged part", f"{sc.cost_per_check:.2f} USD (36 s at 30 USD/h)",
         "only uncertain parts are sent; capacity is planned"]])}
<h4>Why a normal defect classifier is not enough</h4>
<ul><li>A classifier (“good / scratch / dent / …”) needs <b>hundreds of labelled images per defect type</b>. Defects are
rare (≈1% of parts), so collecting them takes months.</li>
<li>New defect types appear (a new tool wears differently, a new supplier) — a classifier has never seen them and lets
them pass.</li>
<li>Labelling pixel masks is expensive and inconsistent between people.</li></ul>
<h4>The turnaround: learn “normal”, flag the unusual</h4>
<p><b>Anomaly detection</b> models only the good parts, which every factory has in abundance. Anything that does not look
like the good parts — whatever kind of defect it is — gets a high score. The price: the model cannot name the defect
type, and anything unusual (including a dimmer lamp) looks suspicious; chapters 11–13 handle that.</p>
<h4>Why three decisions and not two</h4>
<p>Scores of good and defective parts overlap a little. Instead of forcing a guess in that overlap, a narrow
<b>CHECK band</b> around the threshold routes uncertain parts to a human: cheaper than scrapping them, safer than shipping
them. The band is sized so that inspectors are not overloaded (chapter 11).</p>
{callout("key", "The status quo DefectSpotter is compared with",
         f"<p>100% manual inspection of {sc.parts_per_shift:,.0f} parts per 8-hour shift needs "
         f"<b>{manual['inspectors_needed']:.0f} inspectors</b>, still misses <b>{manual['defects_escaping']:.0f} of "
         f"{manual['defective']:.0f}</b> defects (people catch ≈ {100 * sc.human_detection_rate:.0f}% of what they see) "
         f"and costs about <b>{manual['cost_per_shift']:,.0f} USD per shift</b> — see chapter 17.</p>")}"""
    return chapter("problem", 2, "The problem", body,
                   "Inspection is an economic decision under uncertainty: two kinds of mistakes with very different costs, "
                   "plus a limited number of people.")


# ===================================================================================================== 3. flows
def ch_flows(c) -> str:
    d, n = c.d, c.n
    lat = d.latency[d.latency.method == "patchcore"].set_index("category")
    s = lat.loc["screw"] if "screw" in lat.index else lat.iloc[0]
    sc = c.scenario
    build = steps([
        ("Data — <code>index</code>, <code>eda</code>",
         "Every MVTec image and mask is listed and split into four roles <i>before</i> anything is trained (no leakage). The EDA "
         "stage measures brightness, contrast, sharpness and defect sizes and draws the montages of chapter 4."),
        ("Defects without labels — <code>synthetic</code> (+ optional diffusion script)",
         "Two synthetic defects are painted on every held-out good part (cut-paste, scratch or stain, each with an exact mask). "
         "Optionally <code>generate_diffusion_defects.py</code> refines 12 per product with Stable Diffusion."),
        ("Detectors — <code>patchcore</code>, <code>vae</code>, <code>fvae</code>, <code>mean</code>",
         "PatchCore searches its per-product recipe on held-out good + synthetic defects and builds the memory bank; the two "
         "VAEs are trained from scratch on good parts; the baseline computes aligned mean/std images."),
        ("Stress sets — <code>severity</code>, <code>diffusion</code>",
         "Every synthetic defect is replayed at 50% and 25% strength (the detection limit) and the diffusion defects are scored.",
         ),
        ("Calibrate — <code>calibrate</code>",
         "For every product × detector: 0–100 score scale, pixel scale, ensemble weight, capability check and the cost-optimal "
         "threshold and CHECK band. The ONNX backbone is exported and verified."),
        ("Evaluate — <code>evaluate</code>, <code>requalify</code>, <code>latency</code>, <code>drift</code>, <code>few_shot</code>",
         "All numbers of chapters 12–16 on held-out real defects: AUROC with bootstrap intervals, operating rates, lot-shift test, "
         "speed, drift stress test, new-product experiment."),
        ("Package — <code>examples</code>, <code>summary</code>",
         "Sample images for the app and <code>reports/summary.json</code>. The app reads only <code>artifacts/</code>, "
         "<code>reports/</code> and <code>examples/</code>."),
    ])
    inspect = steps([
        ("Image in", "Any RGB image (camera, upload, API). It is kept at full resolution for the display."),
        ("Camera-health check", "Brightness, contrast and sharpness are compared with the qualified images as z-scores. "
         "|z| &gt; 4 on any signal means the image is outside what the model was qualified on: whatever the model says, the "
         "part goes to CHECK with the message “check lamp / lens / camera”."),
        ("Pose (screws only)", "The part mask (Otsu threshold on the colour distance to the background) gives the principal "
         "axis; the image is rotated so the screw lies horizontally, head on the left, centred."),
        ("Resize + normalise", "To the product's resolution (224 px → 28 × 28 patches; 288 px → 36 × 36) and the ImageNet "
         "mean / standard deviation the backbone expects."),
        ("Backbone", "The frozen ResNet-18 (ONNX Runtime) computes layer2 and layer3 feature maps; layer3 is upsampled and "
         "concatenated: one 384-number descriptor per patch.", f"screw {s.stage_cnn_features_ms:.0f} ms"),
        ("Nearest neighbour", f"Every patch descriptor is compared with the {n['bank_range']} vectors of the memory bank; "
         "the distance to the closest one is the patch's anomaly.", f"screw {s.stage_nearest_neighbour_ms:.0f} ms"),
        ("Anomaly map", "The grid of distances is upsampled to the image, smoothed with a Gaussian and (screws) rotated back "
         "to the original pose; a pixel scale maps it to 0–100 (50 = more unusual than 99.9% of good pixels)."),
        ("Image score", "Largest patch distance (bottle) or mean of the top 1–5% of the map (screw, carpet), then mapped to "
         "0–100: 25 = a typical good part, 75 = a typical synthetic defect."),
        ("Decision", "PASS below <i>t − b</i>, CHECK between, REJECT from <i>t + b</i>, with the product's calibrated threshold "
         "<i>t</i> and half-band <i>b</i> (or the lot's re-qualified ones)."),
        ("Result", "Decision card with the reason, the score bar, the heat map with numbered suspected regions (pixel score "
         "&gt; 50), all detectors' opinions, a JSON record; CHECK parts can be sent to the review queue.",
         f"total {s.latency_ms_median:.0f} ms (median)"),
    ])
    calib = steps([
        ("Score the calibration images", "Held-out good parts (never seen by the model), synthetic defects at 100%, 50% and "
         "25% strength and the diffusion defects go through the detector."),
        ("Fit the 0–100 score scale", "25 = median raw score of the good parts, 75 = median of the synthetic defects; linear in "
         "between and beyond. Every detector and product now speaks the same language."),
        ("Capability gate", "A defect set influences the threshold only if the detector separates it from good parts (AUROC ≥ "
         "0.9). Faint 25% replays usually fail the gate — they mark the detection limit instead of dragging the threshold "
         "into the good parts."),
        ("Cost search", "Every threshold 0–100 (step 0.5) × half-band {0, 2.5, …, 15} is priced with the cost model at 1% "
         "defects on kernel-smoothed score distributions; the band may use at most half the inspector capacity (5% of "
         "parts). The cheapest pair is the default policy."),
        ("Store", "Everything goes into <code>artifacts/&lt;product&gt;/calibration.json</code>: scales, policies, health "
         "reference, ensemble weight. The real-defect (“oracle”) threshold is computed the same way for comparison only."),
    ])
    human = steps([
        ("CHECK", "The part's score is inside the band, or the camera was out of range."),
        ("Queue", "Batch CHECKs are queued automatically; single inspections with “Send to review queue”."),
        ("Inspect", "The inspector sees the part, the heat map, the score and the model decision; adds a note."),
        ("Verdict", "“Confirm defect → REJECT” or “Release as good → PASS”."),
        ("Audit", "Time, file, product, detector, score, threshold, model decision, verdict and note go into the audit log "
         "(downloadable CSV). Engineers review it; changes are deliberate releases, never automatic retraining."),
    ])
    monitor = steps([
        ("Gate every image", "Camera-health z-scores (|z| &gt; 4 → CHECK)."),
        ("Chart the stream", "EWMA control charts (λ = 0.2, ±3σ, alarm after 2 consecutive points) on the score and on "
         "brightness / sharpness; the PSI compares this week's score histogram with qualification (&lt; 0.1 stable, 0.1–0.25 "
         "watch, &gt; 0.25 act)."),
        ("Diagnose", "Camera signals moved → lamp, focus or position: fix it and re-verify on qualified parts. Only the scores "
         "of good parts moved → probably a new production lot."),
        ("Test the lot", "Inspect ~10 good parts of the current lot; a one-sided Mann–Whitney U test (p &lt; 0.01) decides "
         "whether they score higher than the qualification's good parts."),
        ("Re-qualify", "If yes: the new parts' patches join the memory bank (10% coreset), their leave-one-out scores become "
         "the good reference, synthetic defects are made from them and scale + threshold are re-fitted. Deployed as a "
         "recorded lot model (chapter 12)."),
    ])
    deploy = steps([
        ("Commit", "Code, trained artifacts (≈ 57 MB) and reports live in the GitHub repository; GitHub Actions runs the tests "
         "on every push."),
        ("Release", "<code>scripts/deploy_hf.py</code> uploads the app, package, UI, artifacts, reports, examples and docs to "
         "the Hugging Face Space."),
        ("Serve", "The Space runs <code>app.py</code> (Gradio 6, ZeroGPU hardware; inference itself is CPU/ONNX). Users get the "
         "web UI and three API endpoints."),
        ("Elsewhere", "The same repository runs locally (<code>python app.py</code>) or in Colab; nothing needs a GPU or "
         "PyTorch at inference time."),
    ])
    body = f"""
<p>Six flows describe everything DefectSpotter does. Each diagram is followed by the same steps in words — what happens,
in which order, and what comes out.</p>
{sub("flow-build", "3.1", "Build flow — from raw images to deployable models")}
{svg.figure(svg.build_pipeline(n), "<b>Seventeen stages</b> in seven phases (<code>python scripts/train.py --stages …</code>). "
            "Phase 2 makes defects without any labels. Results are cached in <code>../defectspotter_work</code>; the long "
            "stages are resumable per product.", ("proc", "store", "ghost"))}
{build}
{callout("what", "What is happening", "<p>Nothing in this flow looks at a real defect to make a choice. Real defects appear "
         "only in the evaluation phase, on images that never trained or calibrated anything. That is what makes the reported "
         "numbers an honest preview of a factory that has not seen its first defect yet.</p>")}
{sub("flow-inspect", "3.2", "Inspection flow — one part, end to end")}
{svg.figure(svg.inspection_flow(n), f"<b>Ten steps with the deployed detector (PatchCore), ≈ "
            f"{lat.latency_ms_median.min():.0f}–{lat.latency_ms_median.max():.0f} ms</b> per part on a 4-thread CPU. The dashed arrow "
            "is the camera-health override. With another detector, steps 5–6 change: the feature-space VAE rebuilds the ResNet "
            "features instead of searching the bank; the pixel VAE and the mean baseline skip the ResNet entirely "
            "(chapter 10).", ("data", "model", "proc", "gate"))}
{inspect}
{table(["product", "backbone (ms)", "nearest neighbour (ms)", "total median (ms)", "p95 (ms)", "parts per minute"],
       [[k, f"{lat.loc[k].stage_cnn_features_ms:.0f}", f"{lat.loc[k].stage_nearest_neighbour_ms:.0f}",
         f"{lat.loc[k].latency_ms_median:.0f}", f"{lat.loc[k].latency_ms_p95:.0f}", f"{lat.loc[k].throughput_parts_per_min:.0f}"]
        for k in d.cats if k in lat.index], num_cols=(1, 2, 3, 4, 5),
       caption="PatchCore timing per stage (reports/latency.csv). Screw is slower: 288 px input (36 × 36 patches) and pose "
               "normalisation. The total includes decoding, heat map and overlay rendering.")}
{sub("flow-calibrate", "3.3", "Calibration flow — choosing thresholds without real defects")}
{svg.figure(svg.calibration_flow(n), "<b>From scores to a policy.</b> Real defects (dashed) are scored the same way only to "
            "show how close the label-free threshold lands to the one labels would give.", ("data", "model", "proc", "gate", "store", "ghost"))}
{calib}
{sub("flow-human", "3.4", "Human-in-the-loop flow — what happens to a CHECK")}
{svg.figure(svg.human_flow(n), "<b>Every CHECK is decided by a person and recorded.</b>", ("gate", "human", "store"))}
{human}
{sub("flow-monitor", "3.5", "Monitoring and lot-change flow — after deployment")}
{svg.figure(svg.monitoring_flow(n), "<b>Two different causes, two different fixes.</b> A camera problem is fixed at the camera; "
            "a new lot is handled by re-qualification from good parts.", ("data", "proc", "gate", "alarm", "human", "model"))}
{monitor}
{sub("flow-deploy", "3.6", "Deployment flow — from repository to users")}
{svg.figure(svg.deployment_flow(n), "<b>One repository, three ways to run it.</b>", ("store", "proc", "model", "human", "data"))}
{deploy}"""
    return chapter("flows", 3, "How it works — the flows", body,
                   "What happens, in which order, from raw images to a decision on the line — and after deployment.")


# ===================================================================================================== 4. data
def ch_data(c) -> str:
    d, n = c.d, c.n
    comp = d.composition
    roles = d.roles()
    rows = []
    for cat in ["bottle", "screw", "carpet", "metal_nut"]:
        g = comp[comp.category == cat]
        if g.empty:
            continue
        train = int(g[(g.split == "train") & (g.defect_type == "good")].images.sum())
        tgood = int(g[(g.split == "test") & (g.defect_type == "good")].images.sum())
        tdef = g[(g.split == "test") & (g.defect_type != "good")]
        why = {"bottle": "glass object, fixed top view — the “easy” case",
               "screw": "small metal part at <b>random orientation</b> — the hard object",
               "carpet": "woven <b>texture</b> — no object shape to rely on",
               "metal_nut": "held back as a <b>new product</b> for few-shot onboarding"}[cat]
        types = ", ".join(t.replace("_", " ") for t in tdef.defect_type)
        rows.append([f"<b>{cat.replace('_', ' ')}</b>", why, f"{train}", f"{tgood}", f"{int(tdef.images.sum())}", types])
    role_rows = [[cat.replace("_", " "), *[f"{int(roles.loc[cat, r])}" for r in
                                          ("model_train_good", "calibration_good", "oracle_calibration", "operating_test")]]
                 for cat in roles.index]
    montages = "".join(fig(c.figure(f"eda_montage_{cat}.jpg", 1000), f"<b>{cat.replace('_', ' ')}</b> — good parts (top) and "
                                                                       "defects with their masks", f"{cat} montage")
                       for cat in ["bottle", "screw", "carpet", "metal_nut"])
    locs = "".join(fig(c.figure(f"eda_defect_location_{cat}.png", 400), f"{cat.replace('_', ' ')}", f"defect locations {cat}")
                   for cat in ["bottle", "screw", "carpet", "metal_nut"])
    body = f"""
<p><b>MVTec AD</b> (MVTec Software GmbH; Bergmann et al., IJCV 2021) is the standard benchmark for industrial anomaly
detection: 5,354 high-resolution images of 15 products. Its <b>training set contains only good parts</b> — exactly the
situation of a factory. The <b>test set</b> contains good parts and real defects, each defect with a pixel-accurate
<b>ground-truth mask</b>. The project uses four products:</p>
{table(["product", "why it was chosen", "train (good)", "test good", "test defective", "defect types"], rows, num_cols=(2, 3, 4))}
<p class="tbl-note">Licence <b>CC BY-NC-SA 4.0</b> (research and teaching, no commercial use). Downloaded from the Hugging Face
mirror <code>{e(d.manifest['dataset']['mirror'])}</code> pinned to revision <code>{e(d.manifest['dataset']['revision'][:12])}…</code>,
so every run gets byte-identical files (<code>scripts/download_data.py</code>).</p>
{sub("data-eda", "4.1", "What the exploratory analysis showed")}
<div class="grid2">{montages}</div>
<ul><li><b>Tiny defects dominate.</b> Most defect masks cover less than 1% of the image — localisation metrics matter
and down-scaling can hide defects (one reason screws use 288 px).</li>
<li><b>Acquisition is tightly controlled</b> (small brightness / contrast spread). That makes good-only learning possible
— and makes a lamp or focus change dangerous (chapter 13).</li>
<li><b>Test captures differ slightly from training captures.</b> Carpet test images are brighter and come from a different
roll; screw test images are darker. This “lot change” is real and is handled in chapter 12.</li>
<li><b>Test sets are ≈ 70% defective</b>, unlike a real line (≈ 1%). All rates are therefore reported per class and costs
are re-weighted to 1% defects.</li></ul>
<div class="grid4">{locs}</div>
<p class="small">Where defects occur (sum of all defect masks per product, reports/figures/eda_defect_location_*.png):
bottle defects sit on the rim and inside; screw defects anywhere along the part; carpet defects anywhere.</p>
{sub("data-split", "4.2", "The leakage-free split")}
{svg.figure(svg.split_flow(n), "<b>Four roles.</b> The 20% of training goods held out for calibration play the part of "
            "“good parts the model has never seen”; the test set is split by defect type so both halves contain every type.",
            ("data", "model", "proc", "gate", "ghost"))}
{table(["product", "① model_train_good", "② calibration_good", "③ oracle_calibration", "④ operating_test"], role_rows,
       num_cols=(1, 2, 3, 4), caption="Images per role (reports/splits.csv, seed 42).")}
{callout("why", "Why this matters", "<p>If the same images chose the threshold <i>and</i> measured it, the reported escape "
         "rate would be optimistic. Here the deployed thresholds come from ② + synthetic defects only, and every reported "
         "operating number comes from ④, which nothing ever saw before evaluation.</p>")}"""
    return chapter("data", 4, "The data", body, "Where the images come from, what they look like and how they are split.")


# ===================================================================================================== 5. synthetic
def ch_synthetic(c) -> str:
    d = c.d
    syn = d.synthetic.groupby(["category", "synthetic_type"]).mask_area_pct.agg(["count", "median"]).reset_index()
    syn_rows = [[r.category, r.synthetic_type, f"{int(r['count'])}", f"{r['median']:.2f}%"] for _, r in syn.iterrows()]
    prompts = d.diffusion.groupby(["category", "synthetic_type"]).agg(prompt=("prompt", "first"),
                                                                          change=("mean_change_in_mask", "mean"),
                                                                          seconds=("seconds", "mean")).reset_index()
    p_rows = [[r.category, r.synthetic_type.replace("diffusion_", ""), f"<i>{e(r.prompt)}</i>", f"{r.change:.0f}",
               f"{r.seconds:.0f}"] for _, r in prompts.iterrows()]
    cap = d.capability[d.capability.method == "patchcore"]
    cap_rows = [[r.category, r.calibration_set, f"{r.synthetic_auroc:.3f}", f"{int(r.n)}",
                 "✓ used" if bool(r.used) else "— detection limit only"] for _, r in cap.iterrows()]
    examples = []
    for cat in d.cats:
        for kind in ("cutpaste", "scratch", "stain"):
            p = c.root / "examples" / "synthetic" / f"{cat}_{kind}_0.jpg"
            if p.exists():
                examples.append((c.thumb(p), f"{cat} · {kind}"))
    diff_examples = []
    for cat in d.cats:
        for kind in ("cutpaste", "scratch", "stain"):
            p = c.root / "examples" / "diffusion" / f"{cat}_diffusion_{kind}_0.jpg"
            if p.exists():
                diff_examples.append((c.thumb(p), f"{cat} · SD {kind}"))
    body = f"""
<p>To choose a threshold you need to know what a defect's score looks like — but there are no defect images. The project
<b>paints its own defects</b> on held-out good parts. They are not meant to look exactly like real defects; they only need
to be <i>unusual in a similar way</i> (local, small, with a different texture or brightness).</p>
{sub("syn-kinds", "5.1", "Procedural defects (three kinds)")}
{table(["kind", "how it is made", "random parameters"], [
        ["<b>cut-paste</b><br><span class='small'>(CutPaste, Li et al. 2021)</span>",
         "a patch of the same part is cut out, rotated, re-coloured and pasted elsewhere on the part with soft edges",
         "patch size h/40–h/7 × w/40–w/6; rotation ±45°; contrast ×0.6–1.5; brightness ×0.65–1.35; opacity 0.6–0.95"],
        ["<b>scratch</b>", "a thin jittered poly-line, darker (65%) or brighter (35%) than the surface",
         "length 1/10–1/2 of the image side; thickness 1/300–1/90; 3–6 jittered points"],
        ["<b>stain</b>", "an irregular noise-shaped blob with a colour / brightness shift and soft edges",
         "size, shape, colour shift and opacity drawn per defect"]])}
<p>Each defect comes with its exact mask and is kept <b>on the part</b>: a foreground mask (Otsu threshold on the colour
distance to the background) restricts where defects may appear — a scratch on the empty tray would teach nothing. Two
defects are made per held-out good part ({len(d.synthetic):,} in total):</p>
{table(["product", "kind", "defects", "median mask area"], syn_rows, num_cols=(2, 3))}
{thumbs(examples)}
{sub("syn-severity", "5.2", "Severity replays — the detection limit")}
<p>Every synthetic defect is replayed at <b>50%</b> and <b>25%</b> strength (alpha-blended with the original). Full-strength
defects are easy; the faint replays show where detection stops, like the minimum defect size in an inspection
specification. They also make the recipe search meaningful (chapter 7): on full-strength defects alone every setting
scores ≈ 1.0.</p>
{sub("syn-diffusion", "5.3", "Diffusion-refined defects (generative AI)")}
<p>A <b>diffusion model</b> generates images by learning to remove noise step by step. <b>Stable Diffusion 1.5</b> with
<b>SDEdit</b> (image-to-image: add partial noise — <i>strength</i> 0.45 — to a crop around a procedural defect and let the
model denoise it, guided by a product-specific prompt) keeps the defect's position and shape but gives it photographic
texture. An <b>LCM-LoRA</b> (a small low-rank adapter that distils the model to work in a handful of steps) makes this
feasible on a CPU: <b>6 steps, guidance 1.5, ≈ 17 s per image</b>. Only the masked region is composited back, so the
rest of the part stays pixel-identical. 12 per product (4 per kind).</p>
{table(["product", "kind", "prompt", "mean change in mask (grey levels)", "seconds"], p_rows, num_cols=(3, 4))}
{thumbs(diff_examples)}
{sub("syn-capability", "5.4", "Capability check — which defect sets may set the threshold")}
<p>A defect set only influences the default threshold if the detector can actually separate it from good parts
(synthetic AUROC ≥ 0.9). Otherwise the cost search would push the threshold down into the good parts to catch defects the
detector cannot see — and create false rejects. For the deployed PatchCore:</p>
{table(["product", "defect set", "AUROC vs held-out good", "n", "used for the threshold?"], cap_rows, num_cols=(2, 3))}"""
    return chapter("synthetic", 5, "Defects without labels", body,
                   "Synthetic and diffusion-generated defects replace real defect images wherever a choice has to be made.")


# ===================================================================================================== 6. backbone
def ch_backbone(c) -> str:
    d = c.d
    fe = d.manifest["feature_extractor"]
    b = d.backbone[d.backbone.image_size == 224].copy()
    b["best"] = b[["image_auroc_max", "image_auroc_topk"]].max(axis=1)
    g = b.groupby("backbone").agg(auroc=("best", "mean"), pixel=("pixel_auroc", "mean"), dim=("feature_dim", "first"),
                                  onnx=("onnx_mb_fp32_est", "first"), ms=("extract_ms_per_image", "mean")).sort_values("auroc")
    names = {"mobilenet_v3_small": "MobileNetV3-Small", "efficientnet_b0": "EfficientNet-B0", "resnet18": "ResNet-18",
             "wide_resnet50_2": "WideResNet-50-2"}
    bb_rows = [[f"<b>{names.get(k, k)}</b>" if k == "resnet18" else names.get(k, k), f"{r.auroc:.3f}", f"{r.pixel:.3f}",
                f"{int(r.dim)}", f"{r.onnx:.1f}", f"{r.ms:.0f}"] for k, r in g.iterrows()]
    body = f"""
{sub("bb-concepts", "6.1", "Concepts")}
<ul><li><b>Convolutional neural network (CNN).</b> A stack of small learned filters slid over the image. Early layers respond
to edges and colours, middle layers to textures and small shapes, deep layers to object parts. Each layer outputs a
<b>feature map</b>: a grid in which every cell holds a vector describing the image patch under it.</li>
<li><b>ResNet.</b> A CNN built from <b>residual blocks</b> that add their input to their output (<code>y = x + F(x)</code>).
The shortcut lets very deep networks train. <b>ResNet-18</b> has 18 weight layers in four stages (<code>layer1</code>–
<code>layer4</code>, 64 / 128 / 256 / 512 channels); each stage halves the resolution (stride 4, 8, 16, 32).</li>
<li><b>Transfer learning.</b> The weights come from <b>ImageNet-1k</b> (1.28 million everyday photos, 1,000 classes). Those
generic visual features describe industrial surfaces remarkably well.</li>
<li><b>Frozen = not fine-tuned.</b> Not a single backbone weight changes and <b>no classification head</b> is attached:
there are no defect labels to train one, and fine-tuning on good parts alone could collapse the features. The backbone is
a fixed “feature camera”.</li></ul>
{sub("bb-exact", "6.2", "Exactly what is used")}
{table(["item", "value"], [
        ["weights", "torchvision ResNet-18, ImageNet-1k"],
        ["layers kept", f"<code>conv1 → layer1 → layer2 → layer3</code> — {fe['parameters']:,} parameters; <code>layer4</code>, global "
                        "pooling and the classifier head are removed (too ImageNet-specific, not needed)"],
        ["feature maps", "<code>layer2</code> (128 channels, stride 8 — fine texture) + <code>layer3</code> (256 channels, stride 16 — "
                         "object parts) upsampled to layer2's grid and concatenated → <b>384 numbers per patch</b>"],
        ["two outputs", "<code>patch_features</code>: 3 × 3 neighbourhood average (PatchCore's default, tolerant to tiny shifts) · "
                        "<code>patch_features_p1</code>: raw descriptors (keep fine texture; all three deployed recipes use them)"],
        ["input", "224 px → 28 × 28 grid, 288 px → 36 × 36 grid; ImageNet mean / std normalisation"],
        ["export", f"ONNX opset {fe['opset']}, dynamic batch and spatial size, {fe['size_mb']:.1f} MB, verified against PyTorch "
                   f"(max abs error {fe['max_abs_error_vs_torch']:.1e})"],
        ["runtime", "ONNX Runtime on CPU — the app needs neither PyTorch nor a GPU"],
        ["used by", "<b>PatchCore</b> (memory bank of these descriptors) and the <b>feature-space VAE</b> (learns to rebuild them) — "
                    "the pixel VAE and the mean-image baseline work on raw pixels and never use the backbone"]])}
{sub("bb-study", "6.3", "Why ResNet-18 — the backbone study")}
<p>All four candidates are networks pre-trained on ImageNet (the photo collection, not a model); none of them is
re-trained. Each was plugged into PatchCore as the frozen feature extractor at 224 px on the three products
(<code>scripts/backbone_study.py</code>, analysis only) to see which one describes industrial surfaces best per megabyte:</p>
{table(["backbone", "mean image AUROC", "mean pixel AUROC", "feature dim", "ONNX size (MB, est.)", "ms per image"], bb_rows,
       num_cols=(1, 2, 3, 4, 5), hl_rows=[list(g.index).index("resnet18")] if "resnet18" in g.index else ())}
{callout("why", "Why", f"<p>WideResNet-50 is slightly more accurate but its ONNX file is ≈ {g.loc['wide_resnet50_2', 'onnx']:.0f} "
         "MB and it is ≈ 2× slower. ResNet-18 is the most accurate backbone inside a 10–20 MB deployment budget, and the "
         "per-product recipe (chapter 7) closes most of the remaining gap.</p>" if "wide_resnet50_2" in g.index else "")}"""
    return chapter("backbone", 6, "The backbone: a frozen ResNet-18", body,
                   "One pre-trained network turns every image into a grid of patch descriptors; it is never re-trained.")


# ===================================================================================================== 7. patchcore
def ch_patchcore(c) -> str:
    d, n = c.d, c.n
    cfg = d.summary["patchcore_configs"]
    rec_rows = []
    for cat in d.cats:
        k, sel = cfg[cat], d.selected_recipe(cat)
        rec_rows.append([f"<b>{cat}</b>", f"{k['image_size']} px", "pose-normalised" if k.get("align") else "as captured",
                         "raw" if int(k["pool_size"]) == 1 else "3 × 3 average", f"{100 * k['coreset_ratio']:.0f}%",
                         f"{int(sel.bank_vectors):,}", f"{sel.bank_mb:.1f}", f"{k['blur_sigma']:.0f}",
                         "max" if k["score_mode"] == "max" else f"top-{100 * k['topk_fraction']:.0f}%", f"{sel.objective:.4f}"])
    space = []
    h = d.hpo
    for col, label in (("image_size", "input resolution (px)"), ("pool_size", "aggregation (1 = raw, 3 = 3 × 3)"),
                       ("align", "pose normalisation"), ("bank_rotations", "rotated copies in the bank"),
                       ("coreset_ratio", "coreset ratio"), ("blur_sigma", "blur σ (px)"), ("score_mode", "score mode")):
        if col in h:
            vals = {cat: sorted(set(map(str, h[h.category == cat][col]))) for cat in d.cats}
            space.append([label, *[", ".join(vals[cat]) for cat in d.cats]])
    abl = []
    for cat in d.cats:
        hh = h[h.category == cat]
        for col, label in (("image_size", "resolution"), ("pool_size", "aggregation"), ("align", "pose")):
            best = hh.groupby(col).objective.max()
            if len(best) > 1:
                abl.append([cat, label, " · ".join(f"{k}: {v:.4f}" for k, v in best.items())])
    pose_items = []
    for p in sorted((c.root / "demo_images" / "screw").glob("*.png"))[:6]:
        img = c.read_rgb(p)
        aligned, _ = c.align(img, "screw")
        pose_items.append((c.thumb(img), f"captured · {p.stem[:18]}"))
        pose_items.append((c.thumb(aligned), "pose-normalised"))
    body = f"""
{sub("pc-idea", "7.1", "The idea in one sentence")}
<p>A part is normal if <b>every small patch of it looks like some patch of some good part</b>. PatchCore (Roth et al.,
CVPR 2022) stores the patch descriptors of good parts in a <b>memory bank</b> and scores a new part by how far its patches
are from their nearest stored neighbours.</p>
{svg.figure(svg.patchcore_flow(n), "<b>Build once, query for every part.</b> Numbers are bottle / screw / carpet.",
            ("data", "model", "proc", "gate", "store"))}
{sub("pc-math", "7.2", "The mathematics")}
{formula('''patch descriptor of a new part at grid cell (i, j):   q_ij  (384 numbers)
memory bank:                                          M = {{m_1, …, m_K}}
patch anomaly:          s_ij = min_k ‖ q_ij − m_k ‖₂             (Euclidean distance to the nearest normal patch)
anomaly map:            A = GaussianBlur( upsample(s), σ )      (σ = 2 or 4 px)
image score:            max_ij s_ij                (score mode "max")
                        mean of the largest k% of A (score mode "top-k", k = 1% or 5%)''')}
<p>The nearest-neighbour search is exact: distances are computed as <code>‖q‖² − 2 q·m + ‖m‖²</code> in chunks of 8,192
bank vectors with NumPy matrix products — fast enough for a few thousand vectors without an index library.</p>
{sub("pc-coreset", "7.3", "Coreset — a small bank that still covers everything")}
<p>Hundreds of good images × 784–1,296 patches = {n['patch_counts']} vectors (bottle / screw / carpet). Searching all of
them would be slow and the file large. A <b>greedy k-center coreset</b> keeps a small subset that still <i>covers</i> them:</p>
{code('''selected = [random patch]
distance_to_selected = distance of every patch to it
repeat until K patches are selected:
    p = the patch with the largest distance_to_selected      # the least covered patch
    selected.append(p)
    distance_to_selected = min(distance_to_selected, distance to p)''', "text")}
<p>Rare-but-normal details (a reflection seen in 2% of bottles) survive, because they are far from everything else —
random subsampling would drop them. For speed, the distances inside the loop use a random <b>Johnson–Lindenstrauss
projection</b> to 128 dimensions (which approximately preserves distances). Two engineering tricks: one greedy run serves
every coreset ratio (its first <i>K</i> picks <i>are</i> the size-<i>K</i> coreset), and very large sets are first reduced to a
random 300,000-vector candidate pool. Banks are stored in float16.</p>
{sub("pc-recipe", "7.4", "The per-product recipe, chosen without labels")}
<p>Products differ: a bottle top needs a smooth view, carpet needs fine texture, screws need resolution and a canonical
pose. Every combination below was tried per product and scored on <b>held-out good parts + synthetic defects</b>:</p>
{table(["option", *d.cats], space)}
{formula("objective = 0.4 × image AUROC + 0.3 × TPR at ≤ 1% false rejects + 0.3 × pixel AUROC   (full-strength + 50% synthetic defects)")}
<p><b>Parsimony rule:</b> among all settings within <b>0.005</b> of the best objective (differences that small are sampling
noise), deploy the one with the cheapest nearest-neighbour search. The deployed recipes:</p>
{table(["product", "input", "pose", "aggregation", "coreset", "bank vectors", "bank MB", "blur σ", "score", "objective"],
       rec_rows, num_cols=(5, 6, 7, 9))}
{details("What each design choice bought (best objective per option)", table(["product", "choice", "best objective per value"], abl))}
{sub("pc-pose", "7.5", "Pose normalisation for screws")}
<p>CNN features are not rotation-invariant, and screws lie on the tray at any angle, so a memory bank would need every
orientation of every thread. Classic machine vision fixes the pose first:</p>
{steps([("Mask", "Otsu threshold on the colour distance to the background → the part's pixels."),
        ("Axis", "PCA of the mask pixels: the first principal axis is the screw's long axis; elongation = √(λ₁/λ₂) ≈ 5–7 for "
                 "screws, ≈ 1 for the round bottle top (never aligned)."),
        ("Head left", "The wider end (the head) is detected from the mask width at both ends and placed on the left."),
        ("Centre", "The part's <i>extent</i> (not its centroid, which sits near the heavy head) is moved to the image centre, "
                   "so long screws are never cut off."),
        ("Rotate", "One affine warp; corners are filled with the background colour. Rotation keeps the thread's handedness "
                   "(a mirror flip would not)."),
        ("Warp back", "Heat maps are computed on the aligned image and warped back to the original frame (pixel-centre "
                      "convention), so overlays and pixel metrics are unaffected.")])}
{thumbs(pose_items)}
{callout("why", "Why it mattered", f"<p>Screw image AUROC went from 0.935 (v1) to 0.983 with the per-product recipe (v2) to "
         f"<b>{d.metric('screw', 'patchcore'):.3f}</b> with pose normalisation (v3); screw pixel AUROC "
         f"{d.metric('screw', 'patchcore', 'pixel_auroc'):.3f}.</p>")}
{callout("code", "In the code", "<p><code>defectspotter/patchcore.py</code> (bank, coreset, scoring), <code>defectspotter/pose.py</code>, "
         "<code>defectspotter/pipeline.py</code> → <code>stage_patchcore</code>, <code>_patchcore_search</code>, <code>_select_patchcore</code>.</p>")}"""
    return chapter("patchcore", 7, "PatchCore — a memory of normal patches", body,
                   "The deployed detector: no training by gradient descent, just a well-chosen memory of normal patches.")


# ===================================================================================================== 8. VAEs
def ch_vae(c) -> str:
    d, n = c.d, c.n

    def hist_rows(h):
        g = h.groupby("category").agg(epochs=("epoch", "max"), best=("val_loss", "min"), seconds=("seconds", "sum"))
        return [[cat, f"{int(g.loc[cat, 'epochs'])}", f"{g.loc[cat, 'best']:.4f}", f"{g.loc[cat, 'seconds'] / 60:.1f}"]
                for cat in d.cats if cat in g.index]
    weights = {cat: d.cal[cat]["methods"]["ensemble"].get("weight_patchcore") for cat in d.cats}
    body = f"""
{callout("key", "Two different VAEs, two different inputs",
         "<p>The <b>pixel VAE</b> (v1) learns to rebuild the raw image — no ResNet involved. The <b>feature-space VAE</b> (v3) "
         "learns to rebuild the <b>output of the frozen ResNet-18</b> (the 384 × 28 × 28 feature map) — so its path is "
         "image → ResNet → VAE. Both are trained from scratch on good parts only; the ResNet itself is never trained.</p>")}
{sub("vae-concepts", "8.1", "Autoencoders and VAEs — the concepts")}
<ul><li>An <b>autoencoder</b> has an <b>encoder</b> that compresses the input into a small code (the <b>latent</b>) and a
<b>decoder</b> that rebuilds the input from it. It is trained to make the reconstruction equal to the input.</li>
<li>Trained <b>only on good parts</b>, it learns to rebuild <i>normal</i> appearance. Shown a defect, it rebuilds what a good
part would look like there — so the <b>reconstruction error</b> lights up where the defect is.</li>
<li>A <b>variational</b> autoencoder (VAE, Kingma &amp; Welling 2014) makes the latent random: the encoder outputs a mean μ and a
log-variance log σ² per latent number, and a sample <code>z = μ + σ·ε</code> with ε ~ N(0, 1) is decoded (the
<b>reparameterisation trick</b>, which keeps the sampling differentiable).</li>
<li>A <b>KL-divergence</b> term keeps the latent distribution close to a standard normal. This regularises the code (smooth,
no memorising of individual images) and makes the model generative — you could sample new good-looking parts.</li></ul>
{formula('''loss = reconstruction error(x, x̂) + β · KL( q(z|x) ‖ N(0, I) )
KL   = −½ · Σ ( 1 + log σ² − μ² − σ² )          (summed over latent numbers, averaged over the batch)
β    = 0.001   (small, so reconstructions stay sharp)
at test time the mean μ is decoded (deterministic); the anomaly map is the reconstruction error''')}
{svg.figure(svg.vae_flow(n), "<b>Same idea, different inputs.</b> The pixel VAE rebuilds 49,152 pixel values from 128 numbers; "
            "the feature VAE rebuilds the backbone's 384 × 28 × 28 descriptor grid from a 32 × 7 × 7 spatial latent.",
            ("data", "model", "proc", "gate"))}
{sub("vae-pixel", "8.2", "Pixel VAE (v1) — kept for comparison and for its reconstruction view")}
{table(["item", "value"], [
        ["input", "RGB image resized to 128 × 128, values in [0, 1]"],
        ["encoder", "4 stages of (stride-2 4 × 4 conv → GroupNorm → SiLU → 3 × 3 conv → GroupNorm → SiLU), 32 → 64 → 128 → 128 "
                    "channels, 1 × 1 squeeze to 32 channels (8 × 8)"],
        ["latent", "128 numbers (fully connected from the 32 × 8 × 8 map): μ and log σ²"],
        ["decoder", "mirror image with transposed convolutions, sigmoid output"],
        ["loss", "L1 + 0.5 · (1 − SSIM) + 0.001 · KL — <b>SSIM</b> (structural similarity) compares local brightness, contrast "
                 "and structure, so blurry averages are not rewarded"],
        ["training", "120 epochs, AdamW (weight decay 1e-5), one-cycle learning rate up to 1e-3, batch 32; augmentation: "
                     "horizontal flips (not for screws — it would mirror the thread), 90° rotations, ±4% brightness; best "
                     "validation checkpoint kept"],
        ["anomaly map", "0.5 · |input − reconstruction| + 0.5 · (1 − SSIM map), Gaussian σ 2; score = mean of the top 1%"],
        ["size", "≈ 2 M parameters, 7.4 MB ONNX per product"]])}
{table(["product", "epochs", "best validation loss", "training minutes (CPU)"], hist_rows(d.vae_hist), num_cols=(1, 2, 3))}
<p>Squeezing a whole image through 128 numbers gives a <b>blurry average part</b>: the error is large everywhere, so defects
barely stand out (screw image AUROC {d.metric('screw', 'vae'):.3f} — no better than chance).</p>
{sub("vae-feature", "8.3", "Feature-space VAE (v3) — the generative detector")}
<p>The better VAE reconstructs <b>what the backbone sees</b> instead of raw pixels: the 384 × 28 × 28 grid of patch
descriptors (224 px input, 3 × 3-averaged — the same features PatchCore uses).</p>
{table(["item", "value"], [
        ["input", "backbone feature map, normalised per channel with the training mean / std (std floored at 10% of the median "
                  "channel std, so near-constant channels cannot explode the error)"],
        ["encoder", "1 × 1 conv 384 → 256 · 3 × 3 stride-2 conv 256 → 128 (14 × 14) · 3 × 3 stride-2 conv 128 → 128 (7 × 7); "
                    "GroupNorm + SiLU"],
        ["latent", "<b>spatial</b>: 32 channels × 7 × 7 (μ and log σ² per position) — keeps <i>where</i> things are"],
        ["decoder", "1 × 1 conv → two 4 × 4 stride-2 transposed convolutions back to 28 × 28 → 1 × 1 conv to 384 channels"],
        ["loss", "mean-squared error on the normalised features + 0.001 · KL"],
        ["training", "60 epochs, AdamW, one-cycle learning rate up to 2e-3, batch 16, feature-map flips (not for screws); best "
                     "validation checkpoint kept"],
        ["anomaly map", "per-patch squared reconstruction error (computed inside the ONNX graph), upsampled, Gaussian σ 4; "
                        "score = mean of the top 1%"],
        ["size", "1.44 M parameters, 5.5 MB ONNX per product (+ the shared backbone)"]])}
{table(["product", "epochs", "best validation loss", "training minutes (CPU)"], hist_rows(d.fvae_hist), num_cols=(1, 2, 3))}
{callout("why", "Why the feature VAE works so much better",
         f"<p>Descriptors are compact and semantic — a scratch is “unusual texture”, not thousands of changed pixel values — "
         f"and the spatial latent keeps the layout. Normal patches are reconstructed accurately; defect patches, which the "
         f"decoder never learned to produce, are not. Mean image AUROC: pixel VAE {d.macro('vae'):.3f} → feature VAE "
         f"<b>{d.macro('fvae'):.3f}</b>. It still trails PatchCore on screws ({d.metric('screw', 'fvae'):.3f}): a VAE "
         "generalises, so it partly rebuilds unusual thread patterns too, while a memory bank cannot.</p>")}
{sub("vae-ensemble", "8.4", "Ensemble")}
{formula("ensemble score = w × PatchCore score + (1 − w) × feature-VAE score        (both on the 0–100 scale)")}
<p><i>w</i> is chosen on held-out good parts + synthetic defects (full and 50% strength), scanning from PatchCore-only
downwards so that ties keep the stronger single detector: w = {" / ".join(f"{float(weights[k]):.1f}" for k in d.cats)}
({" / ".join(d.cats)}). Chapter 10 shows it ties with PatchCore and is therefore not deployed.</p>
{callout("code", "In the code", "<p><code>defectspotter/vae.py</code>, <code>defectspotter/fvae.py</code>; stages <code>vae</code>, "
         "<code>fvae</code> and the ensemble weight in <code>stage_calibrate</code> (<code>defectspotter/pipeline.py</code>).</p>")}"""
    return chapter("vae", 8, "Generative AI: variational autoencoders", body,
                   "Two VAEs learn to rebuild good parts; whatever they cannot rebuild is suspicious.")


# ===================================================================================================== 9. baseline
def ch_baseline(c) -> str:
    d = c.d
    rows = [[cat, d.mean_cfg[cat]["alignment"], f"{d.mean_cfg[cat]['image_size']} px", f"{d.mean_cfg[cat]['std_floor']:g}",
             f"{d.mean_cfg[cat]['blur_sigma']:g}", f"{d.metric(cat, 'mean'):.3f}"] for cat in d.cats]
    figs = "".join(fig(c.figure(f"baseline_mean_std_{cat}.jpg", 900), f"<b>{cat}</b>: mean image and standard-deviation image of "
                                                                      "the aligned good parts", f"baseline {cat}") for cat in d.cats)
    body = f"""
<p>The classical machine-vision reference: every image is aligned to a median good reference with OpenCV's <b>ECC</b>
algorithm (enhanced correlation coefficient; none / translation / rigid / affine chosen on synthetic data), then compared
pixel by pixel with the <b>mean and standard deviation</b> of the aligned good parts.</p>
{formula("z(x, y) = | pixel(x, y) − mean(x, y) | / max( std(x, y), floor )      → Gaussian blur → score = mean of the top 1%")}
{table(["product", "alignment chosen", "input", "std floor", "blur σ", "image AUROC"], rows, num_cols=(3, 4, 5))}
<div class="grid3">{figs}</div>
{callout("why", "Why keep it", "<p>It is transparent, fast and needs no deep learning — and it shows exactly where learned "
         "features are needed: it works on the fixed bottle view but fails on rotating screws (a pixel-wise mean of rotated "
         "screws is a blur) and on textures, where every pixel varies.</p>")}"""
    return chapter("baseline", 9, "Classical baseline: the aligned mean image", body,
                   "What a classical vision engineer would build first — the yardstick for the learned detectors.")


# ===================================================================================================== 10. comparison
def ch_compare(c) -> str:
    from ui import explain as ex
    from defectspotter.constants import METHOD_LABELS
    stats = c.detector_stats.set_index("method")
    cats = c.d.cats
    qual = []
    for m in stats.index:
        p = ex.DETECTOR_PROFILES[m]
        qual.append([f"<b>{e(METHOD_LABELS.get(m, m))}</b>", p["family"], p["learns"], p["scores"], p["strength"], p["weakness"]])
    best = {col: ex._best(stats[col], hi) for col, hi in
            [*[(f"auroc_{k}", True) for k in cats], ("auroc_mean", True), ("pixel_auroc", True), ("aupro", True),
             ("cost_per_1000", False)]}

    def cell(m, col, txt):
        return f"<b>★ {txt}</b>" if m in best[col] else txt
    quant = []
    for m in stats.index:
        s = stats.loc[m]
        quant.append([f"<b>{e(METHOD_LABELS.get(m, m))}</b>", *[cell(m, f"auroc_{k}", f"{s[f'auroc_{k}']:.3f}") for k in cats],
                      cell(m, "auroc_mean", f"{s.auroc_mean:.3f}"), cell(m, "pixel_auroc", f"{s.pixel_auroc:.3f}"),
                      cell(m, "aupro", f"{s.aupro:.3f}"), pct(s.false_reject_rate), pct(s.escape_rate), pct(s.check_rate),
                      cell(m, "cost_per_1000", f"{s.cost_per_1000:,.0f}"), f"{s.latency_ms:.0f}",
                      f"<b>{ex.DETECTOR_PROFILES[m]['verdict']}</b>"])
    pc, ens, fv, vae, mean = (stats.loc[m] for m in ("patchcore", "ensemble", "fvae", "vae", "mean"))
    from types import SimpleNamespace
    from ui.store import ReportStore
    per_product = ex.detector_per_product(ReportStore(c.root), SimpleNamespace(categories=cats))
    body = f"""
<p>Every entry of the app's <b>Detector</b> menu is a separate, fully built and calibrated model. Each learned <b>only from
good parts</b>, got its own 0–100 score scale and its own label-free PASS / CHECK / REJECT thresholds, and is evaluated on
the same held-out real defects. They differ in <i>what they remember about a good part</i> and <i>how they measure
“unusual”</i>.</p>
{sub("cmp-differ", "10.1", "How they differ")}
{svg.figure(svg.detector_map(c.n), "<b>Five separate paths.</b> Only ① and ② use the frozen ResNet-18.",
            ("data", "model", "best", "proc", "gate"))}
{table(["detector", "family", "what it learns from good parts", "how it scores a new part", "strength", "weakness"], qual)}
{sub("cmp-numbers", "10.2", "How well they work — and which one is better")}
{table(["detector", *[f"AUROC {k}" for k in cats], "mean AUROC", "pixel AUROC", "AUPRO 30%", "false rejects", "escapes",
        "to CHECK", "cost / 1,000 parts (USD)", "ms / part", "verdict"], quant,
       num_cols=tuple(range(1, 4 + len(cats) + 6)),
       caption="★ = best in the column (ties within 0.0005). Operating columns: each detector at its own label-free default "
               "threshold on the operating test split, mean of the products, qualification-lot models (PatchCore's carpet "
               "CHECK share falls from 49.6% to 5.7% after re-qualification). Cost: expected cost per 1,000 parts at 1% "
               "defects (escape 400, false reject 2, check 0.30 USD).")}
{sub("cmp-products", "10.3", "Which detector is best for which product?")}
{table(["product", "best ranking (image AUROC)", "best heat map (pixel AUROC · AUPRO)", "lowest cost / 1,000 parts",
        "better VAE (feature vs pixel)", "use for this product"],
       [[f"<b>{e(r['product'])}</b>", e(r['ranking']), e(r['heat map']), e(r['cost']), e(r['generative']),
         f"<b>🏆 {e(r['recommended'])}</b><br><span class='small'>{e(r['why'])}</span>"] for r in per_product],
       caption="“=” marks a tie (within 0.0005). For bottle the ensemble weight is 1.0 — it is PatchCore — so they tie everywhere. "
               "Costs: each detector at its own label-free threshold, qualification-lot models.")}
{callout("key", "Overall verdict: PatchCore is the best detector and the deployed one",
         f"<p><b>PatchCore</b> has the best or tied-best image AUROC on every product (mean {pc.auroc_mean:.3f}), the lowest "
         f"expected cost ({pc.cost_per_1000:,.0f} USD per 1,000 parts) and no false rejects at its label-free threshold. The "
         f"<b>ensemble</b> is a statistical tie (mean {ens.auroc_mean:.3f}; pixel metrics within "
         f"{max(abs(ens.pixel_auroc - pc.pixel_auroc), abs(ens.aupro - pc.aupro)):.3f}) but runs two models "
         f"({ens.latency_ms:.0f} vs {pc.latency_ms:.0f} ms) and costs slightly more ({ens.cost_per_1000:,.0f} USD) — no "
         f"benefit. Of the generative models the <b>feature-space VAE</b> ({fv.auroc_mean:.3f}) is far ahead of the "
         f"<b>pixel VAE</b> ({vae.auroc_mean:.3f}) — on every product. The <b>mean-image baseline</b> ({mean.auroc_mean:.3f}) "
         "shows why learned features are needed. <b>Per product:</b> PatchCore for bottle (tied with the ensemble, which is "
         "PatchCore there), PatchCore for screw (pose-normalised; the VAEs fail on rotation) and PatchCore re-qualified for the "
         "current lot for carpet. The ensemble's only edge — heat-map metrics on screw and carpet, by ≤ 0.002 — is a statistical "
         "tie that does not pay for running two models.</p>")}
<div class="grid2">{fig(c.shot("compare_models"), "<b>Models &amp; training tab</b> — “The five detectors”: how they differ.",
                        "detector comparison in the app", "shot")}
{fig(c.shot("compare_results"), "<b>Results tab</b> — “Model comparison — which detector is best?”.",
     "model comparison table in the app", "shot")}</div>
{callout("tip", "See it live", "<p>In <b>🎬 Live demo → Inspect a part</b>, the gallery “All detectors on this part” shows the "
         "five scores and decisions for the same image; the pixel VAE's reconstruction is shown next to it.</p>")}"""
    return chapter("compare", 10, "The five detectors compared", body,
                   "All five were built and calibrated the same way — so they can be compared fairly. PatchCore wins.")


CHAPTERS = [("Part I · Understand the project", [ch_overview, ch_problem, ch_flows]),
            ("Part II · Data and models", [ch_data, ch_synthetic, ch_backbone, ch_patchcore, ch_vae, ch_baseline, ch_compare])]
