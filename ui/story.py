"""HTML for the presentation sections. Every number is read from reports/, never hard-coded."""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from defectspotter.constants import (CATEGORY_INFO, METHOD_LABELS, MVTEC_CITATION, MVTEC_HF_REPO, MVTEC_HF_REVISION,
                                     MVTEC_LICENSE, MVTEC_OFFICIAL_URL)
from defectspotter.decision import CostModel, reference_costs

from .store import ReportStore
from .theme import STATUS, STATUS_BG, STATUS_ICON, STATUS_TEXT


def _pct(value, digits: int = 1) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{100 * value:.{digits}f}%"


def _num(value, digits: int = 3) -> str:
    return "—" if value is None or not np.isfinite(value) else f"{value:.{digits}f}"


def macro(store: ReportStore, method: str, metric: str) -> float:
    metrics = store.csv("metrics.csv")
    if metrics.empty:
        return float("nan")
    return float(metrics[metrics.method == method][metric].mean())


def cost_summary(store: ReportStore) -> dict:
    ops = store.csv("operating_metrics.csv")
    model = store.json("cost_model.json") or {**CostModel().to_dict(), **reference_costs(CostModel())}
    ours = float("nan")
    if not ops.empty:
        ours = float(ops[(ops.method == "patchcore") & (ops.policy == "synthetic_cost")].expected_cost_per_1000.mean())
    return {"ours": ours, "manual": model.get("manual_100pct_per_1000", float("nan")),
            "none": model.get("no_inspection_per_1000", float("nan")), "model": model}


def hero_html(store: ReportStore, categories: list[str]) -> str:
    summary = store.summary
    latency = store.csv("latency.csv")
    pc_lat = float(latency[latency.method == "patchcore"].latency_ms_median.median()) if not latency.empty else float("nan")
    gap = summary.get("mean_abs_threshold_gap", {}).get("patchcore", float("nan"))
    costs = cost_summary(store)
    saving = 1 - costs["ours"] / costs["manual"] if np.isfinite(costs["ours"]) and costs["manual"] else float("nan")
    kpis = [
        (_num(macro(store, "patchcore", "image_auroc")), "PatchCore image AUROC", "mean of 3 products"),
        (_num(macro(store, "patchcore", "pixel_auroc")), "PatchCore pixel AUROC", "defect localisation"),
        (_num(macro(store, "vae", "image_auroc")), "VAE image AUROC", "generative comparison"),
        (f"{pc_lat:.0f} ms" if np.isfinite(pc_lat) else "—", "per part on CPU", "upload → decision"),
        (f"{summary.get('feature_extractor_mb', float('nan')):.1f} MB", "ONNX feature extractor",
         f"{summary.get('artifact_total_mb', float('nan')):.0f} MB all models"),
        (_pct(saving, 0) if np.isfinite(saving) else "—", "lower cost than manual inspection",
         f"threshold gap synthetic vs real: {gap:.1f} pts" if np.isfinite(gap) else "illustrative cost model"),
    ]
    tiles = "".join(f'<div class="ds-kpi"><div class="v">{html.escape(v)}</div><div class="l">{html.escape(l)}</div>'
                    f'<div class="s">{html.escape(s)}</div></div>' for v, l, s in kpis)
    chips = "".join(f'<span class="ds-chip">{c}</span>' for c in (
        "Trained on good parts only", "PatchCore memory bank · ResNet-18 ONNX", "Convolutional VAE (GenAI)",
        "Synthetic + diffusion defects set the threshold", "PASS · CHECK · REJECT", "Drift monitor", "Few-shot onboarding"))
    products = ", ".join(categories)
    return f"""
<div class="ds-hero">
  <h1>🔎 DefectSpotter</h1>
  <p class="tag"><b>Visual inspection without defect labels.</b> A quality inspector at the end of the line uploads a
  part image (or a batch); DefectSpotter compares it with what <i>good</i> parts look like, returns an anomaly score,
  a heat map of the suspected defect and a <b>PASS / CHECK / REJECT</b> decision — trained without a single defect image.
  Products: <b>{html.escape(products)}</b>.</p>
  {chips}
  <div class="ds-kpis">{tiles}</div>
</div>"""


def section(title: str, body: str) -> str:
    return f'<div class="ds-card"><div class="ds-section-title">{title}</div>{body}</div>'


def problem_html() -> str:
    return section("1 · The business problem", """
<div class="ds-grid2">
 <div>
  <p>Factories rarely have many images of defects: most parts are good, defects are rare, varied and keep changing.
  A supervised classifier needs hundreds of labelled examples <i>per defect type</i> — a new scratch pattern it has
  never seen simply passes.</p>
  <p>Two errors cost money in opposite directions:</p>
  <ul>
   <li><b>Escape</b> (missed defect): a defective part reaches the customer → complaints, warranty, recalls.</li>
   <li><b>False reject</b>: a good part is scrapped or reworked → material, throughput and inspector time.</li>
  </ul>
 </div>
 <div>
  <div class="ds-step"><b>User.</b> A quality inspector at the end of a production line.</div>
  <div class="ds-step"><b>Idea.</b> Learn the appearance of <i>good</i> parts only; flag anything unusual and
  <i>show where</i> it is.</div>
  <div class="ds-step"><b>Output.</b> Anomaly score (0–100), heat map, suspected regions and a PASS / CHECK /
  REJECT decision; CHECK goes to a human.</div>
  <div class="ds-step"><b>Operating point.</b> A threshold slider exposes the trade-off between false rejects and
  escapes; the default is chosen with <i>synthetic</i> defects and an explicit cost model.</div>
 </div>
</div>""")


def pipeline_html() -> str:
    flow = [
        ("Good parts only", "train/good images per product"),
        ("Frozen CNN", "ResNet-18 layer2+3 → 384-d patch features (ONNX)"),
        ("Memory bank", "greedy coreset of normal patches (fp16)"),
        ("Nearest neighbour", "distance of every patch to normal → heat map"),
        ("Calibrate", "held-out good + synthetic defects → 0–100 scale & cost-optimal threshold"),
        ("Decide", "PASS · CHECK (human) · REJECT"),
        ("Monitor", "camera health + EWMA drift alarm"),
    ]
    boxes = '<span class="arrow">→</span>'.join(f'<div class="box"><b>{a}</b>{b}</div>' for a, b in flow)
    return section("2 · How DefectSpotter works", f"""
<div class="ds-flow">{boxes}</div>
<div class="ds-grid3" style="margin-top:12px">
 <div class="ds-card"><h3>PatchCore-style detector</h3>
  <p class="ds-note">A frozen ImageNet ResNet-18 turns the image into a 28×28 grid of patch descriptors (layer2 + layer3,
  3×3 neighbourhood averaging). Descriptors of all good training parts form a memory bank, compressed with greedy k-center
  <b>coreset</b> selection. A test patch far from every normal patch is anomalous; the largest distance is the image score.
  No training of the network is needed — onboarding a product takes seconds.</p></div>
 <div class="ds-card"><h3>Convolutional VAE (Generative AI)</h3>
  <p class="ds-note">An encoder–decoder with a 128-d latent bottleneck is trained to reconstruct good parts (L1 + SSIM loss).
  It cannot reproduce what it never saw, so the reconstruction error (pixel + structural) highlights defects. It is our
  second, independent detector and is compared head-to-head with the feature method.</p></div>
 <div class="ds-card"><h3>Classical baseline</h3>
  <p class="ds-note">Images are aligned to a median good reference with OpenCV ECC (translation / rigid / affine chosen
  automatically), then compared with the per-pixel mean and standard deviation of good parts. Transparent and fast — and it
  shows exactly where learned features are needed (rotating screws, textures).</p></div>
</div>
<div class="ds-grid2" style="margin-top:12px">
 <div class="ds-card"><h3>Synthetic defects choose the threshold</h3>
  <p class="ds-note">Held-out good parts are corrupted with procedural <b>cut-paste</b>, <b>scratch</b> and <b>stain</b>
  defects (with exact masks), refined by <b>Stable-Diffusion inpainting</b> for realistic texture. Every synthetic
  defect is also replayed at <b>25% and 50% severity</b> — the "detection limit" a real inspection spec defines — because
  full-strength synthetic defects are far more conspicuous than real thread damage on screws. A set only joins the default
  calibration if the detector reliably separates it from good parts (synthetic AUROC ≥ 0.9); weaker sets document the
  detection limit. With held-out good parts they (1) set the 0–100 score
  scale (25 = typical good, 75 = typical synthetic defect), (2) tune hyper-parameters and (3) pick the cost-optimal
  PASS/CHECK/REJECT thresholds — no real defect is used. Real defects only measure how close that threshold is to an
  oracle chosen with labels.</p></div>
 <div class="ds-card"><h3>Built to mirror a real line</h3>
  <ul class="ds-note">
   <li>Cost model: 1% prevalence, escape ≫ false reject, inspector time per CHECK, review capacity ≤ 10% of parts.</li>
   <li>Acquisition health gate: if brightness / contrast / focus is outside the qualified range, no automatic
       decision is taken - PASS or REJECT becomes CHECK (no blind shipping, no scrapping good parts under bad light).</li>
   <li>Human review queue + audit log for every CHECK; the model never silently retrains.</li>
   <li>Leakage-free splits: model-fit good · calibration good · oracle-threshold test · operating test.</li>
   <li>Few-shot onboarding of a new product (metal nut) from a handful of good parts.</li>
  </ul></div>
</div>""")


def data_html(store: ReportStore) -> str:
    comp = store.csv("dataset_composition.csv")
    rows = ""
    if not comp.empty:
        for category, part in comp.groupby("category", sort=False):
            train = int(part[part.split == "train"].images.sum())
            test_good = int(part[(part.split == "test") & (part.defect_type == "good")].images.sum())
            test_bad = int(part[(part.split == "test") & (part.defect_type != "good")].images.sum())
            types = ", ".join(sorted(t.replace("_", " ") for t in part[part.defect_type != "good"].defect_type.unique()))
            info = CATEGORY_INFO.get(category, {})
            rows += (f"<tr><td><b>{category}</b><br><span class='ds-small'>{html.escape(info.get('line_context', ''))}</span></td>"
                     f"<td>{train}</td><td>{test_good}</td><td>{test_bad}</td><td>{html.escape(types)}</td></tr>")
    return section("3 · Data", f"""
<p class="ds-note"><a href="{MVTEC_OFFICIAL_URL}" target="_blank">MVTec AD</a> — real industrial images with pixel-accurate
defect masks. Licence: <b>{MVTEC_LICENSE}</b>. Downloaded reproducibly from the Hugging Face mirror
<code>{MVTEC_HF_REPO}</code> pinned to commit <code>{MVTEC_HF_REVISION[:8]}</code>. {html.escape(MVTEC_CITATION)}</p>
<table class="ds-table"><tr><th>Product</th><th>Train (good only)</th><th>Test good</th><th>Test defective</th>
<th>Defect types (never used for training)</th></tr>{rows}</table>""")


def results_html(store: ReportStore) -> str:
    metrics = store.csv("metrics.csv")
    thresholds = store.csv("threshold_comparison.csv")
    ops = store.csv("operating_metrics.csv")
    if metrics.empty:
        return section("4 · Results", "<p>Run the training pipeline to populate results.</p>")
    table = "<table class='ds-table'><tr><th>Product</th>" + "".join(
        f"<th>{METHOD_LABELS[m]}</th>" for m in ("patchcore", "vae", "mean", "ensemble")) + "</tr>"
    for category in metrics.category.unique():
        table += f"<tr><td><b>{category}</b></td>"
        for method in ("patchcore", "vae", "mean", "ensemble"):
            row = metrics[(metrics.category == category) & (metrics.method == method)]
            if row.empty:
                table += "<td>—</td>"
                continue
            r = row.iloc[0]
            table += (f"<td>{r.image_auroc:.3f} <span class='ds-small'>img</span> · {r.pixel_auroc:.3f} "
                      f"<span class='ds-small'>px</span> · {r.aupro_30:.3f} <span class='ds-small'>PRO</span></td>")
        table += "</tr>"
    table += "</table>"
    best = metrics.groupby("method").image_auroc.mean().sort_values(ascending=False)
    if best.get("patchcore", 0) >= best.get("vae", 0):
        verdict_title = "Feature memory beats reconstruction"
        verdict_text = ("The VAE reproduces a blurry version of every part, so subtle defects hide inside its own "
                        "reconstruction error; frozen CNN features keep the local detail PatchCore compares.")
    else:
        verdict_title = "Reconstruction is competitive here"
        verdict_text = "On this run the VAE ranks parts better on average; see the per-product table for where."
    pc_ops = ops[(ops.method == "patchcore") & (ops.policy == "synthetic_cost")] if not ops.empty else pd.DataFrame()
    frr = float(pc_ops.false_reject_rate.mean()) if not pc_ops.empty else float("nan")
    esc = float(pc_ops.escape_rate.mean()) if not pc_ops.empty else float("nan")
    rev = float(pc_ops.review_rate.mean()) if not pc_ops.empty else float("nan")
    gap_rows = thresholds[thresholds.method == "patchcore"] if not thresholds.empty else pd.DataFrame()
    gap_text = ", ".join(f"{r.category}: {r.synthetic_threshold:.1f} vs {r.real_threshold:.1f}" for r in gap_rows.itertuples())
    return section("4 · Results on held-out real defects", f"""
<p class="ds-note">Image AUROC (ranking of defective vs good parts), pixel AUROC and AUPRO@30% FPR (localisation) on the
MVTec test images — none of which were used for training or threshold selection.</p>
{table}
<div class="ds-grid3" style="margin-top:12px">
 <div class="ds-card"><h3>{verdict_title}</h3><p class="ds-note">Mean image AUROC:
 {' · '.join(f"{METHOD_LABELS[m]} <b>{v:.3f}</b>" for m, v in best.items())}. {verdict_text}</p></div>
 <div class="ds-card"><h3>Operating point (PatchCore)</h3><p class="ds-note">At the synthetic-only threshold on the
 operating test split: false-reject rate <b>{_pct(frr)}</b>, escape rate <b>{_pct(esc)}</b>, human review load
 <b>{_pct(rev)}</b> of parts (prevalence-adjusted).</p></div>
 <div class="ds-card"><h3>Synthetic vs real threshold</h3><p class="ds-note">Cost-optimal threshold chosen with synthetic
 defects vs with real labelled defects (oracle): {html.escape(gap_text) or '—'}. Explore it in the <b>Threshold &amp; cost</b> tab.</p></div>
</div>""")


def responsible_html() -> str:
    return section("5 · Responsible AI", """
<div class="ds-grid3">
 <div class="ds-card"><h3>Human in the loop</h3><p class="ds-note">CHECK is a first-class outcome, not an error. Every CHECK
 enters the review queue; the inspector's verdict and note are logged with the model score, threshold and version.
 Heat maps show <i>visual deviation</i>, not root cause.</p></div>
 <div class="ds-card"><h3>Lighting &amp; camera drift</h3><p class="ds-note">A dimming lamp, a dirty lens or a bumped camera
 shifts every image. A good-only model reads that as "unusual" and false rejects explode. We monitor camera-health signals
 (brightness, contrast, focus) and the score stream with EWMA control charts; an alarm triggers maintenance and re-qualification,
 never silent retraining.</p></div>
 <div class="ds-card"><h3>Limits &amp; licence</h3><p class="ds-note">The score is a ranking, not a defect probability.
 Synthetic defects are a calibration tool, not proof of coverage — validate every critical defect mode on plant data.
 MVTec AD is CC BY-NC-SA 4.0: research and teaching only.</p></div>
</div>""")


def demo_guide_html() -> str:
    return section("6 · Live demo in 2 minutes", """
<ol class="ds-note">
 <li><b>Inspect</b>: choose a product, click a sample (good or defective) or upload/paste/take a photo — see the score,
 heat map, suspected regions, VAE reconstruction and all detectors side by side; ground truth is shown for samples.</li>
 <li>Move the <b>threshold</b> and <b>CHECK band</b> sliders at the top — decisions update everywhere.</li>
 <li><b>Batch</b>: load the sample batch → summary table, decision mix, CSV export; CHECK parts flow to the
 <b>Review</b> tab where you act as the inspector.</li>
 <li><b>Threshold &amp; cost</b>: change the cost of an escape or the defect rate and apply the new cost-optimal threshold.</li>
 <li><b>Drift</b>: dim the lamp or defocus the lens on a good part and watch false rejects appear — and the
 camera-health gate catch them.</li>
 <li><b>New product</b>: onboard the metal nut from 8 good images and inspect it seconds later.</li>
</ol>""")


def decision_card(decision: str, score: float, threshold: float, band: float, policy: str, latency_ms: float,
                  reasons: list[str] | None = None, truth: str | None = None) -> str:
    color, bg, text = STATUS[decision], STATUS_BG[decision], STATUS_TEXT[decision]
    meaning = {"PASS": "Within the learned envelope of good parts — ship.",
               "CHECK": "Uncertain — route to the quality inspector (review queue).",
               "REJECT": "Clear visual anomaly — quarantine for rework/scrap per plant procedure."}[decision]
    extra = "".join(f"<div class='why'>⚠ {html.escape(r)}</div>" for r in (reasons or []))
    truth_html = f"<div class='why'>Ground truth for this sample: <b>{html.escape(truth)}</b></div>" if truth else ""
    return (f"<div class='ds-decision' style='background:{bg};border-color:{color};color:{text}'>"
            f"<div class='big'>{STATUS_ICON[decision]} {decision}</div>"
            f"<div class='meta'>score <b>{score:.1f}</b> · threshold {threshold:.1f} · CHECK band ±{band:.1f} · "
            f"policy <i>{html.escape(policy)}</i> · {latency_ms:.0f} ms</div>"
            f"<div class='why'>{meaning}</div>{extra}{truth_html}</div>")


def explanation_md(result: dict) -> str:
    score, t, b = result["score"], result["threshold"], result["band"]
    lo, hi = t - b, t + b
    if score < lo:
        logic = f"score {score:.1f} is below the PASS boundary {lo:.1f}"
    elif score < hi:
        logic = f"score {score:.1f} lies inside the human-review band {lo:.1f}–{hi:.1f}"
    else:
        logic = f"score {score:.1f} is above the REJECT boundary {hi:.1f}"
    regions = result["regions"]
    region_text = "no pixel region exceeds the unusual-pixel level (50)" if regions.empty else (
        f"{len(regions)} suspected region(s); largest covers {regions.area_pct.max():.2f}% of the image, peak pixel score "
        f"{regions.peak.max():.0f}")
    others = [c for c in result.get("comparison", []) if c["method"] != result["method"] and np.isfinite(c.get("score", np.nan))]
    agree = sum(c["decision"] == result["decision"] for c in others)
    timing = " · ".join(f"{k.replace('_ms', '').replace('_', ' ')} {v:.0f} ms" for k, v in result["timing"].items())
    return (f"**Why {result['decision']}?** The {METHOD_LABELS[result['method']]} {logic}. "
            f"Localisation: {region_text}. {agree}/{len(others)} other detectors agree.  \n"
            f"**Timing:** {timing}.  \n"
            "_The score is a normalised anomaly ranking (25 ≈ typical good part, 75 ≈ typical synthetic defect), "
            "not a probability. The heat map shows visual deviation, not root cause._")
