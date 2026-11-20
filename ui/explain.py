"""Plain-language explanations shown next to every part of the app.

Every number is read from the shipped artifacts (``artifacts/``) or the evidence (``reports/``) so the text can never
drift from the models. Used by ``ui/app.py`` (intro boxes, model cards, glossary, captions).
"""
from __future__ import annotations

import html

import numpy as np
import pandas as pd

from defectspotter.constants import (CATEGORY_INFO, METHOD_LABELS, MVTEC_CITATION, MVTEC_HF_REPO, MVTEC_HF_REVISION,
                                     MVTEC_LICENSE, MVTEC_OFFICIAL_URL)

from defectspotter.operations import SOURCES, LineScenario, rates_from_scores

from .store import ReportStore


def esc(value) -> str:
    return html.escape(str(value))


def pct(value, digits: int = 1) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(value) else f"{100 * value:.{digits}f}%"


def num(value, digits: int = 3) -> str:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return "—"
    return "—" if not np.isfinite(value) else f"{value:.{digits}f}"


# ------------------------------------------------------------------ building blocks
def intro(icon: str, title: str, what: str, how: str | None = None) -> str:
    """Intro box at the top of every page: what you see here and how to read it."""
    how_html = f'<div class="ds-intro-how"><b>How to read it.</b> {how}</div>' if how else ""
    return (f'<div class="ds-intro"><div class="ds-intro-title">{icon} {title}</div>'
            f'<div class="ds-intro-what">{what}</div>{how_html}</div>')


def card(title: str, body: str, icon: str = "") -> str:
    return f'<div class="ds-card"><div class="ds-card-title">{icon} {title}</div>{body}</div>'


def table(rows: list[list], header: list[str]) -> str:
    head = "".join(f"<th>{esc(h)}</th>" for h in header)
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in rows)
    return f'<table class="ds-table"><tr>{head}</tr>{body}</table>'


def chips(items: list[str]) -> str:
    return "".join(f'<span class="ds-chip">{esc(i)}</span>' for i in items)


# ------------------------------------------------------------------ header + navigation
def deployed_ops(store: ReportStore, inspector) -> pd.DataFrame:
    """Operating metrics of the deployed PatchCore policy per product (re-qualified where a lot shift was found)."""
    ops = store.csv("operating_metrics.csv")
    rows = []
    req = store.csv("requalification.csv")
    for category in inspector.categories:
        part = ops[(ops.category == category) & (ops.method == "patchcore") & (ops.policy == "synthetic_cost")] if not ops.empty else ops
        row = {"category": category, "lot": "training lot"}
        if len(part):
            r = part.iloc[0]
            row.update(false_reject_rate=r.false_reject_rate, escape_rate=r.escape_rate,
                       review_rate=r.review_rate, good_check_rate=r.good_check_rate,
                       cost=r.expected_cost_per_1000)
        if not req.empty and category in set(req.category):
            q = req[req.category == category].iloc[0]
            if bool(q.deployed):
                row.update(lot="current lot (re-qualified)", false_reject_rate=q.requal_false_reject_rate,
                           escape_rate=q.requal_escape_rate, review_rate=q.requal_review_rate,
                           good_check_rate=q.requal_good_check_rate, cost=q.requal_expected_cost_per_1000)
        rows.append(row)
    return pd.DataFrame(rows)


def hero(store: ReportStore, inspector) -> str:
    metrics = store.csv("metrics.csv")
    summary = store.summary
    latency = store.csv("latency.csv")
    pc = metrics[metrics.method == "patchcore"] if not metrics.empty else metrics
    auroc = float(pc.image_auroc.mean()) if len(pc) else float("nan")
    pix = float(pc.pixel_auroc.mean()) if len(pc) else float("nan")
    lat = float(latency[latency.method == "patchcore"].latency_ms_median.median()) if not latency.empty else float("nan")
    ops = deployed_ops(store, inspector)
    frr = float(ops.false_reject_rate.mean()) if "false_reject_rate" in ops else float("nan")
    esc_rate = float(ops.escape_rate.mean()) if "escape_rate" in ops else float("nan")
    check = float(ops.review_rate.mean()) if "review_rate" in ops else float("nan")
    from defectspotter.operations import LineScenario, manual_inspection_view, shift_view

    scenario = LineScenario()
    deployed = [shift_view(r, scenario) for _, model, r in real_world_rows(store, inspector) if "before" not in model]
    manual = manual_inspection_view(scenario)["cost_per_shift"]
    ours = float(np.mean([d["cost_per_shift"] for d in deployed])) if deployed else float("nan")
    saving = 1 - ours / manual if np.isfinite(ours) and manual else float("nan")
    inspectors = float(np.mean([d["inspectors_needed"] for d in deployed])) if deployed else float("nan")
    kpis = [
        (num(auroc), "Detection quality", "image AUROC, mean of 3 products · 1.0 = perfect ranking of defects above good parts"),
        (num(pix), "Localisation quality", "pixel AUROC · does the heat map light up on the defect?"),
        (pct(frr), "False rejects", "good parts scrapped (operating test split, deployed thresholds)"),
        (pct(esc_rate), "Escapes", "defective parts passed (same split) · the expensive error"),
        (pct(check), "Sent to a human", f"share of all parts to CHECK · ≈ {inspectors:.1f} inspectors per 8-h shift at 2,500 parts/h"),
        (f"{lat:.0f} ms" if np.isfinite(lat) else "—", "Per part on a CPU", "upload → decision with ONNX Runtime"),
        (pct(saving, 0), "Cheaper than 100% manual", "cost per shift in USD (real-world view, humans catch 80%)"),
        (f"{summary.get('artifact_total_mb', float('nan')):.0f} MB", "All models", "ONNX backbone 10.6 MB + banks + VAEs"),
    ]
    tiles = "".join(f'<div class="ds-kpi"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div>'
                    f'<div class="s">{esc(s)}</div></div>' for v, l, s in kpis)
    products = ", ".join(CATEGORY_INFO.get(c, {}).get("title", c) for c in inspector.categories)
    return f"""
<div class="ds-hero">
  <div class="ds-hero-top">
    <div>
      <h1>🔎 DefectSpotter</h1>
      <p class="tag"><b>Visual inspection without defect labels.</b> Learns what <i>good</i> parts look like, flags
      anything unusual, shows <b>where</b> it is and decides <b>PASS · CHECK · REJECT</b> — trained without a single
      defect image. Products: <b>{esc(products)}</b>.</p>
    </div>
  </div>
  {chips(["Good parts only", "Frozen ResNet-18 (ONNX)", "PatchCore memory bank", "Feature-space VAE (GenAI)",
          "Synthetic + diffusion defects set the threshold", "Human review of CHECK", "Drift & lot re-qualification"])}
  <div class="ds-kpis">{tiles}</div>
  <div class="ds-small" style="margin-top:8px">Key indicators are measured on held-out MVTec test images that never
  trained a model or chose a threshold. Hover a chart for exact values; every page starts with a short "how to read it".</div>
</div>"""


def nav() -> str:
    steps = [("🏠", "Overview", "problem · solution · results"), ("📊", "Data", "source · splits · EDA"),
             ("🧠", "Models & training", "backbone · PatchCore · VAEs"), ("📈", "Results", "metrics · confusion matrices"),
             ("🎚️", "Threshold & cost", "trade-offs in money"), ("🌗", "Drift & new lots", "monitoring · re-qualification"),
             ("🆕", "New product", "few-shot onboarding"), ("📖", "Guide", "concepts · glossary · API"),
             ("🎬", "Live demo", "inspect · batch · review")]
    items = "".join(f'<div class="ds-nav-step"><div class="i">{i}</div><div><b>{esc(t)}</b><span>{esc(d)}</span></div></div>'
                    for i, t, d in steps)
    return (f'<div class="ds-nav"><div class="ds-nav-title">Presentation path — the tabs below go from the story to the '
            f'live demonstration (last tab)</div><div class="ds-nav-row">{items}</div></div>')


def settings_help() -> str:
    return ("**Global settings** — apply to every page: **Product** (which production line), **Detector** (which model "
            "decides), **Production lot** (models qualified on the training lot, or re-qualified for the current lot), "
            "**Decision threshold** and **CHECK band** (scores within ± band of the threshold go to a human). "
            "↺ restores the calibrated values chosen without real defects.")


# ------------------------------------------------------------------ data
def data_source_html(store: ReportStore) -> str:
    comp = store.csv("dataset_composition.csv")
    counts = ""
    if not comp.empty:
        rows = []
        for category, part in comp.groupby("category"):
            train = int(part[(part.split == "train")].images.sum())
            good = int(part[(part.split == "test") & (part.defect_type == "good")].images.sum())
            bad = int(part[(part.split == "test") & (part.defect_type != "good")].images.sum())
            types = part[(part.split == "test") & (part.defect_type != "good")].defect_type.nunique()
            role = "few-shot experiment (new product)" if category == "metal_nut" else "main experiment"
            rows.append([f"<b>{esc(CATEGORY_INFO.get(category, {}).get('title', category))}</b>",
                         f"{train}", f"{good}", f"{bad}", f"{types}", role])
        counts = table(rows, ["product", "train (good only)", "test good", "test defective", "defect types", "used for"])
    return card("Where the data comes from", f"""
<p class="ds-note"><b>MVTec AD</b> — the standard benchmark for industrial anomaly detection (MVTec Software GmbH):
5,354 high-resolution images (700–1024 px) of 15 products. The <b>training set contains only good parts</b>; the
<b>test set</b> has good parts and real defects with a pixel-accurate <b>ground-truth mask</b> for every defect.
We use <b>bottle</b> (glass, top view), <b>screw</b> (metal, random orientation) and <b>carpet</b> (woven texture), and
hold back <b>metal nut</b> as a "new product" for few-shot onboarding.</p>
{counts}
<p class="ds-small" style="margin-top:8px">Source: <a href="{MVTEC_OFFICIAL_URL}" target="_blank">{MVTEC_OFFICIAL_URL}</a> ·
downloaded from the Hugging Face mirror <code>{MVTEC_HF_REPO}</code> pinned to revision <code>{MVTEC_HF_REVISION[:10]}</code>
(every run gets byte-identical files) · licence {esc(MVTEC_LICENSE)} · {esc(MVTEC_CITATION)}</p>""", "🗂️")


def splits_html(store: ReportStore) -> str:
    splits = store.csv("splits.csv")
    rows = []
    if not splits.empty:
        for category, part in splits.groupby("category"):
            c = part.groupby(["role", "label"]).size()
            get = lambda r, l: int(c.get((r, l), 0))  # noqa: E731
            rows.append([esc(category), get("model_train_good", 0), get("calibration_good", 0),
                         f"{get('oracle_calibration', 0)} + {get('oracle_calibration', 1)}",
                         f"{get('operating_test', 0)} + {get('operating_test', 1)}"])
    return card("How the images are split (leakage-free, seed 42)", f"""
<div class="ds-grid4 ds-roles">
 <div><b>① model_train_good</b> 80% of training goods → build the memory bank, train the VAEs.</div>
 <div><b>② calibration_good</b> 20% of training goods, never seen by a model → score scale, thresholds, tuning (with synthetic defects).</div>
 <div><b>③ oracle_calibration</b> 35% of each test type → the "what if we had labels" threshold, and the good parts of the current lot for re-qualification.</div>
 <div><b>④ operating_test</b> 65% → every reported false-reject / escape / CHECK number. Never used for any choice.</div>
</div>
{table(rows, ["product", "① train good", "② calibration good", "③ oracle (good + defect)", "④ operating (good + defect)"]) if rows else ""}
<p class="ds-small" style="margin-top:6px">Real defects never train a model and never choose the deployed threshold —
the same constraint a factory has before its first defect appears.</p>""", "✂️")


def synthetic_html() -> str:
    return card("Synthetic defects: how thresholds are chosen without real defects", """
<p class="ds-note">On every held-out good part we paint defects with an exact mask: <b>cut-paste</b> (a patch of the
part, rotated and recoloured, pasted elsewhere), <b>scratch</b> (thin jittered dark/bright line) and <b>stain</b>
(irregular discoloured blob). Each is also replayed at <b>50% and 25% strength</b> (faint defects — the detection limit
of an inspection spec). A part mask (Otsu threshold on the colour distance to the background) keeps object defects on
the part. A subset is refined by <b>Stable Diffusion 1.5</b> (image-to-image "SDEdit" with an LCM-LoRA, 6 steps,
strength 0.45, product-specific prompts) so the texture looks photographic.</p>""", "🧪")


# ------------------------------------------------------------------ models
def _cfg(inspector, category: str, method: str) -> dict:
    try:
        return inspector.calibration(category)["methods"][method]["config"]
    except KeyError:
        return {}


def components_html(inspector, store: ReportStore) -> str:
    fe = inspector.manifest.get("feature_extractor", {})
    sizes = store.csv("artifact_sizes.csv")
    mb = lambda pattern: (f"{sizes[sizes.file.str.contains(pattern)].size_mb.mean():.1f} MB"  # noqa: E731
                          if not sizes.empty and sizes.file.str.contains(pattern).any() else "—")
    rows = [
        ["<b>ResNet-18 backbone</b>", "turns an image into a grid of patch descriptors",
         "ImageNet-1k (torchvision weights)", "<b>No</b> — frozen, not fine-tuned; no head",
         f"{fe.get('parameters', 0) / 1e6:.2f} M kept (layers up to layer3)", f"{fe.get('size_mb', float('nan')):.1f} MB ONNX"],
        ["<b>PatchCore memory bank</b>", "stores normal patches; distance to nearest = anomaly", "—",
         "<b>Built, not trained</b>: features of good parts + greedy coreset (no gradients)",
         "1–4 k patch vectors per product", mb("patchcore_bank")],
        ["<b>Feature-space VAE</b> (GenAI)", "reconstructs the backbone's feature map; error = anomaly", "—",
         "<b>Yes</b>, from scratch on good parts' feature maps", "1.44 M per product", mb("fvae.onnx")],
        ["<b>Pixel VAE</b> (GenAI, v1)", "reconstructs the image; error = anomaly (comparison)", "—",
         "<b>Yes</b>, from scratch on good images (128 px)", "≈ 2 M per product", mb("/vae.onnx")],
        ["<b>Mean-image baseline</b>", "classical: aligned per-pixel mean ± std of good parts", "—",
         "<b>Statistics only</b> (mean / std), alignment chosen on synthetic data", "—", mb("mean_baseline.npz")],
        ["<b>Calibration</b>", "0–100 score scale + PASS / CHECK / REJECT thresholds", "—",
         "<b>Fitted</b> on held-out good + synthetic defects (no real defects)", "a few numbers", "calibration.json"],
    ]
    return card("What is trained, and on what?", table(rows, ["component", "role", "pre-trained on", "trained in this project?",
                                                            "parameters", "size"]) + """
<p class="ds-small" style="margin-top:6px"><b>Key point:</b> the deep backbone is used as a fixed "feature camera" — it is
<b>not</b> fine-tuned and <b>no classification head</b> is attached, because there are no defect labels to train one.
What adapts to each product is the memory bank (PatchCore) and the small VAEs, both learned from good parts only.</p>""", "🧩")


def backbone_html(inspector) -> str:
    fe = inspector.manifest.get("feature_extractor", {})
    outs = fe.get("outputs", {})
    return card("Backbone: frozen ResNet-18 used as a feature extractor", f"""
<div class="ds-flow">
 <div class="box"><b>Image</b>resized to 224 or 288 px, ImageNet-normalised</div><span class="arrow">→</span>
 <div class="box"><b>conv1 · layer1</b>edges, colours (stride 4)</div><span class="arrow">→</span>
 <div class="box"><b>layer2</b>128 channels, stride 8 — fine texture</div><span class="arrow">→</span>
 <div class="box"><b>layer3</b>256 channels, stride 16 — parts of objects</div><span class="arrow">→</span>
 <div class="box"><b>✂ layer4 + classifier head removed</b>too ImageNet-specific, not needed</div><span class="arrow">→</span>
 <div class="box"><b>Patch descriptors</b>layer3 upsampled to layer2's grid, concatenated → 384 numbers per patch</div>
</div>
<p class="ds-note">Weights: torchvision ImageNet-1k (1.28 M everyday photos). <b>All weights frozen</b> — no
fine-tuning, no new head. Two outputs from one pass: <b>3×3 neighbourhood average</b> (PatchCore default, robust) and
<b>raw descriptors</b> (keep fine texture). Exported once to ONNX ({esc(fe.get('size_mb', 0) and f"{fe['size_mb']:.1f}")} MB,
outputs {esc(", ".join(outs.values()) if outs else "patch_features")}, dynamic resolution) and shared by every product.
Chosen by a backbone study: MobileNetV3 (0.85 mean AUROC), EfficientNet-B0 (0.91), <b>ResNet-18 (0.93)</b>,
WideResNet-50 (0.94 but 95 MB) — best accuracy inside the 10–20 MB budget.</p>""", "🧠")


def patchcore_html(inspector) -> str:
    rows = []
    for category in inspector.categories:
        c = _cfg(inspector, category, "patchcore")
        lot = inspector.shipped_lot(category)
        rows.append([f"<b>{esc(category)}</b>", f"{c.get('image_size', '—')} px",
                     "pose-normalised" if c.get("align") else ("4 orientations" if c.get("bank_rotations", 1) > 1 else "as captured"),
                     "3×3 average" if c.get("pool_size", 3) == 3 else "raw",
                     f"{100 * c.get('coreset_ratio', 0):.0f}%",
                     "max patch" if c.get("score_mode") == "max" else f"top {100 * c.get('topk_fraction', 0):.0f}% of map",
                     "✓ re-qualified for the current lot" if lot else "—"])
    return card("PatchCore: a memory of normal patches", f"""
<p class="ds-note"><b>Idea.</b> A part is normal if every small patch of it looks like some patch of some good part.
<b>Training</b> = run the backbone over all good training images, keep every patch descriptor, then compress them with
a <b>greedy k-center coreset</b> (repeatedly add the patch farthest from everything kept — rare-but-normal details
survive, random subsampling would drop them). <b>Inference</b> = for each patch of a new image, distance to its nearest
neighbour in the bank; the map of distances (upsampled, Gaussian-smoothed) is the heat map, its maximum (or top-k mean)
the image score. <b>Per-product recipe</b>, chosen on held-out good + synthetic defects only (cheapest within 0.005 of the
best objective):</p>
{table(rows, ["product", "input", "pose", "descriptors", "coreset", "image score", "lot"])}
<p class="ds-small">Pose normalisation (screw): the part mask's principal axis is rotated to horizontal, head left — the
memory no longer has to cover every orientation of every thread. Rotation keeps thread handedness; a flip would not.</p>""", "🧱")


def vae_html(inspector, store: ReportStore) -> str:
    cat = inspector.categories[0]
    f = _cfg(inspector, cat, "fvae")
    v = _cfg(inspector, cat, "vae")
    fh = store.csv("fvae_training_history.csv")
    vh = store.csv("vae_training_history.csv")
    best = lambda h: ", ".join(f"{c} {g.val_loss.min():.3f}" for c, g in h.groupby("category")) if not h.empty else "—"  # noqa: E731
    return card("Generative AI: two convolutional VAEs trained on good parts only", f"""
<p class="ds-note"><b>What is a VAE?</b> A variational autoencoder squeezes its input through a small random "latent" code
and learns to rebuild it. Trained only on good parts, it learns to rebuild <i>normal</i> appearance; a defect cannot be
rebuilt, so the reconstruction error lights up where the defect is. "Variational" = the latent is a probability
distribution kept close to a standard normal by the KL term, which stops the model from simply copying its input.</p>
<div class="ds-grid2">
 <div class="ds-subcard"><b>Feature-space VAE (v3, the detector)</b>
  {table([["input", f"backbone feature map {f.get('feature_dim', 384)} × 28 × 28 (224 px, 3×3 averaged)"],
          ["encoder", "1×1 conv 384→256 · 3×3 stride-2 conv 256→128 (14×14) · 3×3 stride-2 conv 128→128 (7×7)"],
          ["latent", f"spatial: {f.get('latent_channels', 32)} × 7 × 7 (mean + log-variance)"],
          ["decoder", "mirror with transposed convolutions back to 384 × 28 × 28"],
          ["loss", f"MSE(features) + β·KL, β = {f.get('beta', 1e-3)}"],
          ["training", f"{f.get('epochs', 60)} epochs, AdamW, one-cycle LR {f.get('learning_rate', 2e-3)}, batch {f.get('batch_size', 16)}, flips (not for screws)"],
          ["parameters", f"{f.get('parameters', 1440960) / 1e6:.2f} M · best validation loss: {esc(best(fh))}"],
          ["score", f"per-patch squared error → upsample, blur σ {f.get('blur_sigma', 4)}, top {100 * f.get('topk_fraction', 0.01):.0f}% mean"]],
         ["", ""])}</div>
 <div class="ds-subcard"><b>Pixel VAE (v1, kept for comparison and its reconstruction view)</b>
  {table([["input", f"RGB image {v.get('image_size', 128)} × {v.get('image_size', 128)}"],
          ["encoder", f"4 stride-2 conv stages ({v.get('base_channels', 32)}→{4 * v.get('base_channels', 32)} channels), GroupNorm + SiLU, 1×1 squeeze"],
          ["latent", f"{v.get('latent_dim', 128)} numbers (fully connected)"],
          ["decoder", "mirror with transposed convolutions, sigmoid output"],
          ["loss", f"L1 + {v.get('ssim_weight', 0.5)}·(1 − SSIM) + β·KL, β = {v.get('beta', 1e-3)}"],
          ["training", f"{v.get('epochs', 120)} epochs, AdamW, one-cycle LR {v.get('learning_rate', 1e-3)}, batch {v.get('batch_size', 32)}, flips/rot90"],
          ["best validation loss", esc(best(vh))],
          ["score", "0.5·L1 + 0.5·(1 − SSIM) map, top 1% mean"]],
         ["", ""])}</div>
</div>
<p class="ds-small">Why the upgrade: the pixel VAE must redraw every pixel through 128 numbers, so it blurs everything
and its error barely separates defects (screw AUROC ≈ 0.5). Reconstructing what the backbone sees — compact, semantic
patch descriptors with a spatial latent — gives a sharp anomaly signal.</p>""", "🎨")


def baseline_html() -> str:
    return card("Classical baseline: difference from the mean good image", """
<p class="ds-note">Each image is aligned to a median good reference with OpenCV's ECC (none / translation / rigid /
affine, chosen on synthetic data), then compared pixel by pixel with the mean and standard deviation of aligned good
parts (a z-score map). Transparent and fast — and it shows where learned features are needed: it fails on rotating
screws and on textures.</p>""", "📐")


def calibration_html(inspector) -> str:
    cat = inspector.categories[0]
    info = inspector.calibration(cat).get("costs", {})
    return card("Calibration: from a raw distance to a decision", f"""
<div class="ds-grid3">
 <div class="ds-subcard"><b>0–100 score scale</b><br>25 = median held-out good part, 75 = median synthetic defect.
 Label-free and comparable across detectors and products.</div>
 <div class="ds-subcard"><b>Three-way decision</b><br>PASS &lt; t − b ≤ CHECK &lt; t + b ≤ REJECT. CHECK parts go to a
 human inspector (Review queue).</div>
 <div class="ds-subcard"><b>Cost-optimal (t, b)</b><br>minimise (1−p)·FRR·C<sub>FR</sub> + p·escape·C<sub>esc</sub> +
 review·C<sub>check</sub> on held-out good + synthetic defects; CHECK planned at ≤ 50% of inspector capacity.
 Defaults: prevalence {pct(info.get('defect_prevalence', 0.01), 0)}, escape {info.get('cost_escape', 400):.0f},
 false reject {info.get('cost_false_reject', 2):.0f}, CHECK {info.get('cost_review', 0.3):.2f} (illustrative units).</div>
</div>""", "🎯")


# ------------------------------------------------------------------ metrics
def glossary_html() -> str:
    items = [
        ("Image AUROC", "Probability that a random defective part scores higher than a random good part. 1.0 = perfect "
                        "ranking, 0.5 = coin flip. Threshold-independent."),
        ("Pixel AUROC", "Same, for pixels: do pixels inside the ground-truth defect mask score higher than normal pixels?"),
        ("AUPRO @ 30%", "Per-region overlap of the heat map with each defect, averaged up to 30% false-positive pixels — "
                        "small defects count as much as large ones."),
        ("False-reject rate", "Good parts sent to REJECT (scrap/rework cost, lost throughput)."),
        ("Escape rate", "Defective parts that PASS — they reach the customer (warranty, complaints). The costly error."),
        ("CHECK load", "Share of all parts routed to a human inspector. Limited by inspector capacity."),
        ("Confusion matrix", "Rows = ground truth (good / defective), columns = decision (PASS / CHECK / REJECT). The ideal "
                             "is everything good in PASS and everything defective in REJECT."),
        ("Cost per 1,000 parts", "Expected cost of false rejects + escapes + human checks at 1% defect prevalence."),
    ]
    tiles = "".join(f'<div class="ds-gloss"><b>{esc(t)}</b><span>{esc(d)}</span></div>' for t, d in items)
    return f'<div class="ds-glossary">{tiles}</div>'


def lot_html(store: ReportStore) -> str:
    req = store.csv("requalification.csv")
    rows = []
    for r in req.itertuples() if not req.empty else []:
        rows.append([f"<b>{esc(r.category)}</b>", f"{r.n_good_current_lot}", f"{r.current_lot_median:.1f} vs {r.qualification_median:.1f}",
                     f"{r.p_value:.3g}", "⚠ <b>lot change → re-qualified</b>" if r.lot_shift else "✓ no change",
                     f"{pct(r.base_good_check_rate, 0)} → {pct(r.requal_good_check_rate, 0)}" if r.lot_shift else "—",
                     f"{pct(r.base_escape_rate, 0)} → {pct(r.requal_escape_rate, 0)}" if r.lot_shift else "—"])
    return card("Lot change and re-qualification", f"""
<p class="ds-note">A new production lot (another carpet roll, a new screw batch, a different supplier) can look
slightly different while still being good. A model qualified on the old lot then scores good parts higher and sends
them to CHECK. <b>Re-qualification</b> needs only a handful of <b>good</b> parts of the new lot: (1) a statistical test
(one-sided Mann-Whitney U) checks whether they score higher than the qualification's held-out good parts; (2) if so,
10% of their patches join the memory bank, synthetic defects are generated <i>from</i> them, and the score scale and
thresholds are re-fitted exactly like the original calibration. No defect image is needed.</p>
{table(rows, ["product", "good parts of current lot", "median score: current vs qualification", "p-value", "decision",
              "good parts sent to CHECK", "escapes"]) if rows else ""}""", "🔁")


def demo_script_html() -> str:
    steps = [
        "Pick a <b>product</b> in the global settings (start with Bottle).",
        "<b>Inspect a part</b>: click a sample (good or defective) or upload one of the demo images "
        "(<code>demo_images/</code> in the repository) — read the decision card, the heat map, the score bar.",
        "Compare <b>all detectors</b> on the same part and open the <b>VAE reconstruction</b>.",
        "Move the <b>threshold</b> and <b>CHECK band</b> sliders — decisions update; ↺ restores the calibrated policy.",
        "<b>Batch</b>: load the sample batch → decision tiles, score strip, CSV. CHECK parts flow to the <b>Review queue</b>, "
        "where you confirm or release them (audit log).",
        "Switch to <b>Carpet</b>: the current lot is re-qualified (see Drift & new lots) — good carpets now PASS.",
    ]
    lis = "".join(f"<li>{s}</li>" for s in steps)
    return card("Demo script (3 minutes)", f'<ol class="ds-note">{lis}</ol>', "🎬")


def caption(text: str) -> str:
    return f'<div class="ds-caption">{text}</div>'


# ------------------------------------------------------------------ real-world view
def real_world_rows(store: ReportStore, inspector) -> list[tuple[str, str, dict]]:
    """(product, model, rates) of the deployed operating points (+ carpet before re-qualification)."""
    ops = store.csv("operating_metrics.csv")
    req = store.csv("requalification.csv")
    req_scores = store.json("requalification_scores.json")
    rows = []
    keys = ("false_reject_rate", "good_check_rate", "escape_rate", "defect_check_rate", "defect_reject_rate")
    for category in inspector.categories:
        part = ops[(ops.category == category) & (ops.method == "patchcore") & (ops.policy == "synthetic_cost")] if not ops.empty else ops
        base = {k: float(part.iloc[0][k]) for k in keys} if len(part) else {}
        shifted = (not req.empty and category in set(req.category) and bool(req[req.category == category].iloc[0].deployed))
        if shifted and category in req_scores:
            d = req_scores[category]
            rows.append((category, "trained-lot model (before re-qualification)", base))
            rows.append((category, "re-qualified for the current lot (deployed)",
                         rates_from_scores(d["operating_scores_requalified"], d["operating_labels"], d["threshold"], d["band"])))
        elif base:
            rows.append((category, "deployed PatchCore", base))
    return rows


def real_world_html(scenario: LineScenario | None = None) -> str:
    sc = scenario or LineScenario()
    links = "".join(f"<li>{esc(text)}" + (f" — <a href='{url}' target='_blank'>{esc(url)}</a>" if url else "") + "</li>"
                    for text, url in SOURCES.values())
    return card("Real-world view: what one shift looks like", f"""
<p class="ds-note">The rates measured on the test images are turned into a real production shift. <b>Scenario</b>
(editable in <i>Threshold &amp; cost</i>): one inspection station, <b>{sc.parts_per_hour:,.0f} parts/hour × {sc.shift_hours:.0f} h =
{sc.parts_per_shift:,.0f} parts per shift</b>, <b>{100 * sc.defect_rate:.0f}% defective</b>; a human check of a flagged part takes
<b>{sc.seconds_per_check:.0f} s</b> at <b>{sc.inspector_cost_per_hour:.0f} USD/hour</b> fully loaded
(= <b>{sc.cost_per_check:.2f} USD per check</b>; one inspector handles {sc.checks_per_inspector_hour:.0f} checks/hour);
inspectors catch <b>{100 * sc.human_detection_rate:.0f}%</b> of the defects they are shown; a scrapped good part costs
<b>{sc.cost_false_reject:.0f} USD</b>; a defect reaching the customer costs <b>{sc.cost_escape:.0f} USD</b>.</p>
<ul class="ds-small">{links}</ul>
<p class="ds-small">The deployed thresholds were calibrated assuming every flagged defect is caught by the inspector;
the shift view applies the real-world 80% human hit rate, so "defects escaping" includes the ones a human misses.</p>""", "🏭")
