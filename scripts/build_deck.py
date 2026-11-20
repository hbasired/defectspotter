"""Build docs/DefectSpotter_Presentation.html — a self-contained 16:9 slide deck for presenting the project.

Every slide states its conclusion in the headline and proves it with one diagram, chart, table or real image. All
numbers come from reports/ and artifacts/ (the same files the app reads). Navigation: ← → / space / PageUp-Down,
Home/End, F = fullscreen, O = overview of all slides; prints one slide per page (Save as PDF).

    python scripts/build_deck.py            # -> docs/DefectSpotter_Presentation.html
"""
from __future__ import annotations

import argparse
import math
import sys
from html import escape
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from build_guide import context  # noqa: E402
from guide_html import helpers, svg  # noqa: E402

COLORS = {"patchcore": "#2a78d6", "fvae": "#6250d6", "vae": "#eb6834", "mean": "#1baf7a", "ensemble": "#c98a00"}
NAMES = {"patchcore": "PatchCore", "ensemble": "Ensemble", "fvae": "Feature-space VAE", "vae": "Pixel VAE",
         "mean": "Mean-image baseline"}
BLUE, GREY = "#2a78d6", "#b9b8b0"


def e(x) -> str:
    return escape(str(x), quote=True)


# --------------------------------------------------------------------------------------------------------- charts
def hbar(items, xmax: float, fmt, width: int = 700, row: int = 46, label_w: int = 210, title: str = "") -> str:
    """Horizontal bars: items = [(label, value, colour, tooltip)]. Value labels at the bar end; tooltips on hover."""
    h = row * len(items) + 34
    plot = width - label_w - 90
    out = [f'<svg viewBox="0 0 {width} {h}" class="chart" role="img" aria-label="{e(title)}"><title>{e(title)}</title>']
    for i in range(5):  # light vertical grid
        x = label_w + plot * i / 4
        out.append(f'<line x1="{x:.1f}" y1="6" x2="{x:.1f}" y2="{h - 26}" class="grid"/>'
                   f'<text x="{x:.1f}" y="{h - 8}" class="tick" text-anchor="middle">{e(fmt(xmax * i / 4))}</text>')
    for i, (label, value, colour, tip) in enumerate(items):
        y = 10 + i * row
        w = max(2.0, plot * float(value) / xmax)
        out.append(f'<text x="{label_w - 12}" y="{y + row / 2 - 2}" class="lab" text-anchor="end">{e(label)}</text>'
                   f'<rect x="{label_w}" y="{y + 6}" width="{w:.1f}" height="{row - 20}" rx="4" fill="{colour}">'
                   f'<title>{e(tip)}</title></rect>'
                   f'<text x="{label_w + w + 8:.1f}" y="{y + row / 2 - 2}" class="val">{e(fmt(value))}</text>')
    return "".join(out) + "</svg>"


def line_chart(points, title: str, width: int = 760, height: int = 380) -> str:
    """Few-shot curve: points = [(shots, auroc, low, high)], log2 x-axis, 0.6–1.0 y-axis, range band + markers."""
    left, right, top, bottom = 70, 24, 20, 56
    xs = [math.log2(p[0]) for p in points]
    x0, x1, y0, y1 = min(xs), max(xs), 0.6, 1.0
    X = lambda v: left + (math.log2(v) - x0) / (x1 - x0) * (width - left - right)  # noqa: E731
    Y = lambda v: top + (y1 - v) / (y1 - y0) * (height - top - bottom)  # noqa: E731
    out = [f'<svg viewBox="0 0 {width} {height}" class="chart" role="img" aria-label="{e(title)}"><title>{e(title)}</title>']
    for v in (0.6, 0.7, 0.8, 0.9, 1.0):
        out.append(f'<line x1="{left}" x2="{width - right}" y1="{Y(v):.1f}" y2="{Y(v):.1f}" class="grid"/>'
                   f'<text x="{left - 10}" y="{Y(v) + 5:.1f}" class="tick" text-anchor="end">{v:.1f}</text>')
    band = " ".join(f"{X(p[0]):.1f},{Y(p[3]):.1f}" for p in points) + " " + \
           " ".join(f"{X(p[0]):.1f},{Y(p[2]):.1f}" for p in reversed(points))
    out.append(f'<polygon points="{band}" fill="{BLUE}" opacity="0.14"/>')
    out.append('<polyline fill="none" stroke="%s" stroke-width="3" points="%s"/>'
               % (BLUE, " ".join(f"{X(p[0]):.1f},{Y(p[1]):.1f}" for p in points)))
    for s, a, lo, hi in points:
        out.append(f'<circle cx="{X(s):.1f}" cy="{Y(a):.1f}" r="7" fill="{BLUE}" stroke="#fff" stroke-width="2">'
                   f'<title>{s} good images: mean AUROC {a:.3f} (range {lo:.3f}–{hi:.3f})</title></circle>'
                   f'<text x="{X(s):.1f}" y="{height - 30}" class="tick" text-anchor="middle">{s}</text>')
        if s in (1, 8, 176):
            out.append(f'<text x="{X(s):.1f}" y="{Y(a) - 16:.1f}" class="val" text-anchor="middle">{a:.3f}</text>')
    out.append(f'<text x="{(left + width - right) / 2:.1f}" y="{height - 6}" class="axis" text-anchor="middle">'
               f'number of good images of the new product (log scale)</text>'
               f'<text x="16" y="{(top + height - bottom) / 2:.1f}" class="axis" text-anchor="middle" '
               f'transform="rotate(-90 16 {(top + height - bottom) / 2:.1f})">image AUROC</text>')
    return "".join(out) + "</svg>"


# --------------------------------------------------------------------------------------------------------- pieces
def tbl(header, rows, num=(), hl=(), cls="") -> str:
    th = "".join(f'<th class="{"num" if i in num else ""}">{h}</th>' for i, h in enumerate(header))
    body = "".join("<tr" + (' class="hl"' if r in hl else "") + ">" +
                   "".join(f'<td class="{"num" if i in num else ""}">{c}</td>' for i, c in enumerate(row)) + "</tr>"
                   for r, row in enumerate(rows))
    return f'<table class="t {cls}"><thead><tr>{th}</tr></thead><tbody>{body}</tbody></table>'


def kpi(value, label, sub="", cls="") -> str:
    return f'<div class="kpi {cls}"><div class="v">{value}</div><div class="l">{label}</div><div class="s">{sub}</div></div>'


def dec(d) -> str:
    return f'<span class="dec {d}">{ {"PASS": "✓", "CHECK": "⚠", "REJECT": "✕"}[d] } {d}</span>'


def zones(t, b, name) -> str:
    lo, hi = t - b, t + b
    return (f'<div class="zrow"><div class="zname">{e(name)}</div><div class="zones" style="grid-template-columns:{lo:g}fr '
            f'{2 * b:g}fr {100 - hi:g}fr"><div class="p">PASS &lt; {lo:g}</div><div class="c">CHECK</div>'
            f'<div class="r">REJECT ≥ {hi:g}</div></div></div>')


def slide(title, sub, body, kind="") -> str:
    head = f'<header><h2>{title}</h2>' + (f'<p class="sub">{sub}</p>' if sub else "") + "</header>"
    return f'<div class="frame"><section class="slide {kind}">{head}<div class="body">{body}</div></section></div>'


def flow_svg(flow) -> str:
    return f'<div class="flow">{flow.svg()}</div>'



def paths_svg(d, stats):
    """Two lanes: detectors on the frozen ResNet-18 vs detectors on raw pixels, each path ending in its accuracy."""
    a = {m: stats.loc[m].auroc_mean for m in stats.index}
    f = svg.Flow(1160, 500, "Which detector uses the ResNet",
                 "Top lane: PatchCore and the feature-space VAE use the frozen ResNet-18. Bottom lane: the pixel VAE and the "
                 "mean-image baseline use raw pixels only.")
    f.lane(168, 6, 982, 240, "USES THE FROZEN RESNET-18  (ImageNet weights — not trained here)")
    f.lane(168, 256, 982, 238, "USES RAW PIXELS — NO RESNET")
    f.node("img", 12, 208, 136, 84, "Part image", ["same image for", "every detector"], "data")
    f.node("rn", 184, 86, 176, 104, "Frozen ResNet-18", ["layer2 + layer3 →", "384 numbers / patch"], "model")
    f.node("px", 184, 330, 176, 104, "Raw pixels", ["128 px (pixel VAE) or", "aligned (baseline)"], "proc")
    rows = [("pc", 40, [("Memory bank", ["good-part patches,", "built — no training"], "store"),
                        ("Nearest neighbour", ["distance per patch"], "proc"), ("Heat map", ["+ 0–100 score"], "proc"),
                        (f"AUROC {a['patchcore']:.3f}", ["① PatchCore", "deployed ★"], "best")], "rn"),
            ("fv", 150, [("Feature VAE", ["trained on good", "parts' features"], "model"),
                         ("Rebuild features", ["error per patch"], "proc"), ("Heat map", ["+ 0–100 score"], "proc"),
                         (f"AUROC {a['fvae']:.3f}", ["② Feature VAE", "best generative"], "model")], "rn"),
            ("pv", 290, [("Pixel VAE", ["trained on good", "images"], "model"),
                         ("Rebuild image", ["error per pixel"], "proc"), ("Heat map", ["+ 0–100 score"], "proc"),
                         (f"AUROC {a['vae']:.3f}", ["③ Pixel VAE", "superseded"], "proc")], "px"),
            ("mb", 400, [("Mean ± std image", ["of aligned good parts"], "proc"),
                         ("z-score", ["per pixel"], "proc"), ("Heat map", ["+ 0–100 score"], "proc"),
                         (f"AUROC {a['mean']:.3f}", ["④ Mean baseline", "reference only"], "proc")], "px")]
    xs, ws = [392, 604, 806, 978], [190, 180, 150, 160]
    for key, y, steps, src in rows:
        ids = []
        for j, (title, sub, kind) in enumerate(steps):
            nid = f"{key}{j}"
            f.node(nid, xs[j], y, ws[j], 76, title, sub, kind)
            ids.append(nid)
        f.edge(src, ids[0])
        for p_, q_ in zip(ids, ids[1:]):
            f.edge(p_, q_)
    f.edge("img", "rn")
    f.edge("img", "px")
    return f


def cards(d, stats, cats):
    spec = [("patchcore", "①", True, "a memory bank of good-part patches (built, not trained)",
             "distance of every patch to its nearest normal patch"),
            ("fvae", "②", True, "a VAE that rebuilds the ResNet features of good parts",
             "how badly the features are rebuilt"),
            ("vae", "③", False, "a VAE that rebuilds good images (128 px)", "how badly the image is rebuilt"),
            ("mean", "④", False, "the mean and spread of aligned good images", "pixel z-score against the mean")]
    out = []
    for m, num, resnet, learns, scores in spec:
        s = stats.loc[m]
        per = " · ".join(f"{k} {s[f'auroc_{k}']:.3f}" for k in cats)
        verdict = {"patchcore": ("best — deployed", "win"), "fvae": ("best generative model", "mid"),
                   "vae": ("superseded", "low"), "mean": ("reference only", "low")}[m]
        out.append(f'''<div class="dcard {verdict[1]}"><div class="dhead" style="border-color:{COLORS[m]}"><span class="num">{num}</span>
<b>{NAMES[m]}</b></div>
<div class="uses {"yes" if resnet else "no"}">{"✓ uses the frozen ResNet-18" if resnet else "✕ raw pixels — no ResNet"}</div>
<div class="row"><span>learns</span>{learns}</div><div class="row"><span>scores by</span>{scores}</div>
<div class="auc"><div class="v">{s.auroc_mean:.3f}</div><div class="track"><div class="fill" style="width:{100 * s.auroc_mean:.1f}%;background:{COLORS[m]}"></div></div>
<div class="per">{per}</div></div>
<div class="foot"><span>{s.latency_ms:.0f} ms / part</span><span class="vd">{verdict[0]}</span></div></div>''')
    return '<div class="dcards">' + "".join(out) + "</div>"


# --------------------------------------------------------------------------------------------------------- slides
def build(c) -> list[str]:
    d, n = c.d, c.n
    from defectspotter.data import read_rgb
    from defectspotter.runtime import Inspector
    from ui import explain as ex
    from ui.store import ReportStore

    store, cats = ReportStore(c.root), d.cats
    stats = c.detector_stats.set_index("method")
    per_product = ex.detector_per_product(store, SimpleNamespace(categories=cats))
    lat = d.latency[d.latency.method == "patchcore"].set_index("category")
    dep = {k: d.deployed(k) for k in cats}
    rates = c.deployed_rates
    pc = stats.loc["patchcore"]
    S = []

    # 1 ── title
    S.append(f'''<div class="frame"><section class="slide title"><div class="tbox">
<div class="eyebrow">Project 3 · Manufacturing · computer vision + generative AI</div>
<h1>DefectSpotter</h1><p class="lead">Visual inspection that learns <b>only from good parts</b>, shows <b>where</b> a part
looks wrong and decides <b>PASS · CHECK · REJECT</b>.</p>
<div class="kpis">{kpi(f"{pc.auroc_mean:.3f}", "detection quality", "mean image AUROC · bottle 1.000 · screw 0.997 · carpet 0.940")}
{kpi("0%", "good parts scrapped", "false rejects at the deployed thresholds")}
{kpi(f"{lat.latency_ms_median.min():.0f}–{lat.latency_ms_median.max():.0f} ms", "per part on a CPU", "ONNX Runtime, no GPU needed")}
{kpi(f"{c.saving_vs_manual:.0f}%", "cheaper per shift", "than an inspector checking every part")}</div>
<div class="links">Live app: <b>bhkr0901-defectspotter.hf.space</b> · Code: <b>github.com/hbasired/defectspotter</b> ·
Data: MVTec AD (CC BY-NC-SA 4.0)</div></div></section></div>''')

    # 2 ── problem
    sc, man = c.scenario, c.manual
    S.append(slide("Defects are rare and expensive — and there are almost no defect images to learn from",
                   "Two errors cost money in opposite directions; today a person looks at every part.", f'''
<div class="cols2"><div>
 <div class="card err"><div class="big">Escape</div><p>a defective part ships to the customer</p><div class="price">{sc.cost_escape:.0f} USD</div></div>
 <div class="card ok"><div class="big">False reject</div><p>a good part is scrapped</p><div class="price">{sc.cost_false_reject:.0f} USD</div></div>
 <div class="card"><div class="big">Human check</div><p>36 s by an inspector</p><div class="price">{sc.cost_per_check:.2f} USD</div></div>
</div><div>
 <h3>Status quo: 100% manual inspection, one 8-hour shift</h3>
 <div class="kpis k3">{kpi(f"{man['inspectors_needed']:.0f}", "inspectors", f"{sc.parts_per_shift:,.0f} parts at 36 s each")}
 {kpi(f"{man['defects_escaping']:.0f} / {man['defective']:.0f}", "defects missed", "people catch ≈ 80% of what they see")}
 {kpi(f"{man['cost_per_shift']:,.0f}", "USD per shift", "checks + escapes")}</div>
 <div class="callout"><b>Why not a defect classifier?</b> It needs hundreds of labelled images <i>per defect type</i> —
 defects are ≈ 1% of parts — and a new defect type passes unseen. <b>DefectSpotter learns “normal” instead.</b></div>
</div></div>'''))

    # 3 ── what it does (real outputs)
    insp = Inspector(c.root / "artifacts")
    bad = insp.inspect(read_rgb(c.root / "demo_images/screw/defect_scratch_head_009.png"), "screw", "patchcore", compare=False)
    good = insp.inspect(read_rgb(c.root / "demo_images/screw/good_025.png"), "screw", "patchcore", compare=False)
    th = helpers.thumb_uri

    def card(r, label):
        return (f'<figure><img src="{th(r["display"], 320)}" alt="{label} input"><figcaption>{label} · input</figcaption></figure>'
                f'<figure><img src="{th(r["overlay"], 320)}" alt="{label} heat map"><figcaption>heat map → {dec(r["decision"])} '
                f'score <b>{r["score"]:.1f}</b></figcaption></figure>')
    S.append(slide("For every part: an anomaly score, a heat map of where, and a decision",
                   "Real outputs of the deployed model on two held-out screws (never seen in training).", f'''
<div class="four">{card(bad, "Scratched head")}{card(good, "Good screw")}</div>
<div class="cols2"><div>
 <h3>How to read it</h3>
 <ul class="tight"><li><b>Score 0–100:</b> 25 = a typical good part, 75 = a typical (synthetic) defect.</li>
 <li><b>Heat map:</b> red where the part differs from everything normal; numbered boxes = suspected regions.</li>
 <li><b>Decision</b> from the product's calibrated threshold ± band:</li></ul></div><div>
 {zones(dep["screw"]["threshold"], dep["screw"]["band"], "screw")}
 <div class="callout">CHECK parts go to a human review queue; every verdict is logged. No defect image was used to
 train the model or to set these thresholds.</div></div></div>'''))

    # 4 ── architecture
    S.append(slide("Architecture: build once per product, inspect every part, keep it honest after deployment",
                   "Three stages; nothing is retrained silently.",
                   flow_svg(svg.system_overview(n)) + svg.legend(("data", "model", "proc", "gate", "human", "store"))))

    # 5 ── which detector uses the ResNet
    S.append(slide("Two families of detectors: two use the ResNet, the other two work on raw pixels",
                   "The same image goes to every detector; each path ends with its accuracy (mean image AUROC, 1.0 = perfect).",
                   flow_svg(paths_svg(d, stats)) + '''
<div class="facts">
 <div><b>The ResNet is never trained.</b> ImageNet weights (torchvision), used frozen as a feature extractor.</div>
 <div><b>ImageNet is data, not a model</b> — 1.28 M photos the weights were learned on.</div>
 <div><b>ResNet → VAE only for ②.</b> The pixel VAE ③ and the baseline ④ never see the ResNet.</div>
 <div><b>⑤ Ensemble</b> = w·① + (1−w)·② — ties with ① but runs two models, so ① is deployed.</div>
</div>'''))

    # 6 ── side-by-side comparison
    S.append(slide("Side by side: features from the ResNet beat raw pixels — PatchCore is best",
                   "Same good-only training data, same label-free calibration, same held-out real defects.",
                   cards(d, stats, cats) + '<p class="note">Bars = mean image AUROC over bottle, screw and carpet (1.0 = every '
                   'defect ranked above every good part). Detailed numbers per product: slides 14–15.</p>'))

    # 6 ── data
    comp, roles = d.composition, d.roles()
    rows = []
    for cat in cats + ["metal_nut"]:
        g = comp[comp.category == cat]
        tdef = g[(g.split == "test") & (g.defect_type != "good")]
        rows.append([f"<b>{cat.replace('_', ' ')}</b>", {"bottle": "fixed top view", "screw": "random orientation",
                     "carpet": "texture", "metal_nut": "new product (few-shot)"}[cat],
                     int(g[(g.split == "train")].images.sum()), int(g[(g.split == "test") & (g.defect_type == "good")].images.sum()),
                     int(tdef.images.sum()), len(tdef)])
    tot = n["roles"]
    S.append(slide("MVTec AD: models learn from good parts only — real defects are used only to measure",
                   "Industrial benchmark with pixel-accurate defect masks; split before anything is trained (no leakage).", f'''
<div class="cols2"><div>{tbl(["product", "challenge", "train (good)", "test good", "test defective", "defect types"], rows, num=(2, 3, 4, 5))}
<div class="split">
 <div class="r1"><b>80% model_train_good</b><span>{tot["model_train_good"]:,} images · build bank, train VAEs</span></div>
 <div class="r2"><b>20% calibration_good</b><span>{tot["calibration_good"]:,} · scale + thresholds (+ synthetic defects)</span></div>
 <div class="r3"><b>35% oracle</b><span>{tot["oracle_calibration"]:,} · comparison only, new-lot goods</span></div>
 <div class="r4"><b>65% operating test</b><span>{tot["operating_test"]:,} · every reported number</span></div>
</div></div>
<div class="imgs1">{"".join(f'<img src="{c.figure(f"eda_montage_{k}.jpg", 760)}" alt="{k} montage">' for k in cats)}</div></div>'''))

    # 7 ── synthetic defects
    syn = [(th(c.root / "examples/synthetic" / f"{k}_{t}_0.jpg", 230), f"{k} · {t}") for k in ("screw", "carpet")
           for t in ("cutpaste", "scratch", "stain")]
    dif = [(th(c.root / "examples/diffusion" / f"{k}_diffusion_{t}_0.jpg", 230), f"{k} · SD {t}") for k in ("screw", "carpet")
           for t in ("cutpaste", "scratch", "stain")]
    grid = lambda items: '<div class="thumbs">' + "".join(f'<figure><img src="{u}" alt="{e(t)}"><figcaption>{e(t)}</figcaption></figure>'  # noqa: E731
                                                         for u, t in items) + "</div>"
    S.append(slide("Thresholds are chosen with painted defects — no real defect image is needed",
                   "Two synthetic defects on every held-out good part, replayed at 50% and 25% strength; 12 per product refined "
                   "by Stable Diffusion.", f'''
<h3>Procedural: cut-paste · scratch · stain (exact masks)</h3>{grid(syn)}
<h3>Generative AI: Stable Diffusion 1.5 SDEdit + LCM-LoRA (6 steps, strength 0.45, ≈ 17 s per image on a CPU)</h3>{grid(dif)}
<div class="callout">A defect set may influence the threshold only if the detector can see it (synthetic AUROC ≥ 0.9) —
faint 25% replays mark the detection limit instead.</div>'''))

    # 8 ── PatchCore
    cfg = d.summary["patchcore_configs"]
    rec = [[f"<b>{k}</b>", f"{cfg[k]['image_size']} px", "yes" if cfg[k].get("align") else "no",
            "raw" if int(cfg[k]["pool_size"]) == 1 else "3×3 avg", f"{100 * cfg[k]['coreset_ratio']:.0f}%",
            f"{int(d.selected_recipe(k).bank_vectors):,}", f"{d.selected_recipe(k).bank_mb:.1f} MB",
            "max" if cfg[k]["score_mode"] == "max" else f"top-{100 * cfg[k]['topk_fraction']:.0f}%",
            f"{d.metric(k, 'patchcore'):.3f}"] for k in cats]
    S.append(slide("Model 1 — PatchCore (deployed): a memory of normal patches, no training",
                   "A part is normal if every patch looks like some patch of a good part; distance to the nearest one = anomaly.",
                   flow_svg(svg.patchcore_flow(n)) +
                   tbl(["product", "input", "pose-normalised", "features", "coreset", "bank vectors", "bank size", "score",
                        "image AUROC"], rec, num=(5, 6, 8)) +
                   '<p class="note">Per-product recipe chosen <b>without real defects</b>: objective = 0.4·AUROC + 0.3·TPR@1% FPR + '
                   '0.3·pixel AUROC on held-out good + synthetic defects; cheapest setting within 0.005 of the best.</p>'))

    # 9 ── pose normalisation
    from defectspotter.pose import align_image
    pairs = []
    for p in sorted((c.root / "demo_images/screw").glob("defect_*.png"))[:4]:
        img = read_rgb(p)
        pairs.append(f'<div class="pp"><img src="{th(img, 220)}" alt="captured"><span>→</span><img src="{th(align_image(img, "screw")[0], 220)}" alt="aligned"></div>')
    hist = [("v1 — one recipe", 0.935, GREY, "screw image AUROC v1 0.935"),
            ("v2 — per-product recipe", 0.983, GREY, "screw image AUROC v2 0.983"),
            ("v3 — pose-normalised", d.metric("screw", "patchcore"), BLUE, f"screw image AUROC v3 {d.metric('screw', 'patchcore'):.3f}")]
    S.append(slide("Screws lie at any angle — rotating each one to the same pose first lifted screw AUROC to 0.997",
                   "PCA axis of the part mask → rotate head-left, centre the part → features → heat map warped back.", f'''
<div class="cols2"><div><h3>captured → pose-normalised</h3><div class="pairs">{"".join(pairs)}</div></div>
<div><h3>Screw image AUROC by version</h3>{hbar(hist, 1.0, lambda v: f"{v:.3f}", title="Screw image AUROC by version")}
<div class="callout">CNN features are not rotation-invariant: without a fixed pose the memory bank must contain every
orientation of every thread.</div></div></div>'''))

    # 10 ── VAEs
    def hist_min(h, k):
        g = h[h.category == k]
        return g.val_loss.min(), g.seconds.sum() / 60
    vae_rows = [["input", "raw image 3×128×128 — <b>no ResNet</b>", "<b>ResNet-18 features</b> 384×28×28"],
                ["latent", "128 numbers (whole image)", "32 × 7 × 7 spatial (keeps layout)"],
                ["loss", "L1 + 0.5·(1−SSIM) + 0.001·KL", "MSE on features + 0.001·KL"],
                ["training", "120 epochs · ≈ 2 M params · 7.4 MB", "60 epochs · 1.44 M params · 5.5 MB"],
                ["anomaly map", "0.5·|x−x̂| + 0.5·(1−SSIM)", "per-patch feature error"],
                ["image AUROC (bottle / screw / carpet)", " / ".join(f"{d.metric(k, 'vae'):.3f}" for k in cats),
                 " / ".join(f"{d.metric(k, 'fvae'):.3f}" for k in cats)],
                ["mean image AUROC", f"{d.macro('vae'):.3f}", f"<b>{d.macro('fvae'):.3f}</b>"]]
    S.append(slide("Models 2 & 3 — two VAEs (generative AI): rebuilding ResNet features beats rebuilding pixels",
                   "A VAE learns to rebuild good parts; what it cannot rebuild is suspicious. Both trained from scratch on good parts.",
                   f'<div class="flow fixed">{svg.vae_flow(n).svg()}</div>' +
                   tbl(["", "③ Pixel VAE (v1) — raw pixels", "② Feature-space VAE (v3) — ResNet features"], vae_rows, hl=(6,), cls="small")))

    # 11 ── baseline + ensemble
    w = " / ".join(f"{float(d.cal[k]['methods']['ensemble']['weight_patchcore']):.1f}" for k in cats)
    S.append(slide("Models 4 & 5 — the classical baseline and the ensemble are references, not winners",
                   "Both are built and calibrated like the others; the comparison shows what they add.", f'''
<div class="cols2"><div class="card big-card"><h3>④ Mean-image baseline (classical)</h3>
<div class="mini">image → OpenCV ECC alignment → |pixel − mean| / max(std, 4) → blur → top-1% mean</div>
<ul class="tight"><li>alignment chosen on synthetic data: {" / ".join(d.mean_cfg[k]["alignment"] for k in cats)}</li>
<li>image AUROC: {" / ".join(f"{d.metric(k, 'mean'):.3f}" for k in cats)} (bottle / screw / carpet)</li>
<li><b>Verdict:</b> fine on the fixed bottle view, fails on rotating screws and on texture → learned features are needed.</li></ul></div>
<div class="card big-card"><h3>⑤ Ensemble (PatchCore + feature VAE)</h3>
<div class="mini">score = w · PatchCore + (1 − w) · feature-space VAE (both on the 0–100 scale)</div>
<ul class="tight"><li>w chosen on held-out good + synthetic defects: {w}</li>
<li>image AUROC: {" / ".join(f"{d.metric(k, 'ensemble'):.3f}" for k in cats)} — mean {stats.loc["ensemble"].auroc_mean:.3f}
vs PatchCore {pc.auroc_mean:.3f}</li>
<li><b>Verdict:</b> a statistical tie that runs two models ({stats.loc["ensemble"].latency_ms:.0f} vs {pc.latency_ms:.0f} ms per part) → not deployed.</li></ul></div></div>'''))

    # 12 ── calibration
    gaps = d.thresholds[d.thresholds.method == "patchcore"].set_index("category")
    S.append(slide("One 0–100 scale and cost-optimal PASS / CHECK / REJECT thresholds — set without real defects",
                   "Same procedure for every product and detector; the labelled-defect threshold is computed only to check it.", f'''
<div class="cols2"><div>
<div class="formula">score = 25 + 50 · (raw − median good) / spread<br><span>25 = typical held-out good part · 75 = typical synthetic defect</span></div>
<div class="formula">cost per part = (1−p)·FRR·2 + p·escapes·400 + checks·0.30<br><span>p = 1% defects · Gaussian-smoothed rates · CHECK ≤ 5% of parts</span></div>
<div class="callout">Grid: threshold 0–100 in steps of 0.5 × CHECK half-band 0–15 → the cheapest pair is deployed.</div></div>
<div><h3>Deployed decision zones</h3>{"".join(zones(dep[k]["threshold"], dep[k]["band"], k + (" (new lot)" if dep[k]["lot"] else "")) for k in cats)}
<h3 style="margin-top:22px">Label-free vs labelled-defect threshold (qualification models)</h3>
{tbl(["product", "label-free", "with real labels", "gap"], [[k, f"{gaps.loc[k, 'synthetic_threshold']:.1f}", f"{gaps.loc[k, 'real_threshold']:.1f}",
     f"{gaps.loc[k, 'gap_synthetic_vs_real']:+.1f}"] for k in cats], num=(1, 2, 3))}</div></div>'''))

    # 13 ── five detectors compared
    order = ["patchcore", "ensemble", "fvae", "vae", "mean"]
    bars = [(NAMES[m], stats.loc[m].auroc_mean, COLORS[m], f"{NAMES[m]}: mean image AUROC {stats.loc[m].auroc_mean:.3f}") for m in order]
    cost = [(NAMES[m], stats.loc[m].cost_per_1000, COLORS[m], f"{NAMES[m]}: {stats.loc[m].cost_per_1000:,.0f} USD per 1,000 parts")
            for m in order]
    rows = [[f'<i class="sw" style="background:{COLORS[m]}"></i><b>{NAMES[m]}</b>',
             *[f"{stats.loc[m][f'auroc_{k}']:.3f}" for k in cats], f"<b>{stats.loc[m].auroc_mean:.3f}</b>",
             f"{stats.loc[m].pixel_auroc:.3f}", f"{100 * stats.loc[m].false_reject_rate:.0f}%", f"{100 * stats.loc[m].escape_rate:.1f}%",
             f"{stats.loc[m].latency_ms:.0f}", ex.DETECTOR_PROFILES[m]["verdict"]] for m in order]
    S.append(slide("PatchCore is the best of the five detectors: highest accuracy and lowest cost",
                   "All five built from good parts only, calibrated the same way, measured on the same held-out real defects.", f'''
<div class="cols2 eq"><div><h3>Mean image AUROC (1.0 = perfect)</h3>{hbar(bars, 1.0, lambda v: f"{v:.3f}", width=680, title="Mean image AUROC")}</div>
<div><h3>Expected cost per 1,000 parts (USD, lower is better)</h3>{hbar(cost, 2000, lambda v: f"{v:,.0f}", width=680, title="Cost per 1,000 parts")}</div></div>
{tbl(["detector", *[f"AUROC {k}" for k in cats], "mean", "pixel AUROC", "false rejects", "escapes", "ms / part", "verdict"], rows,
     num=(1, 2, 3, 4, 5, 6, 7, 8), hl=(0,), cls="small")}'''))

    # 14 ── best per product
    pp = [[f"<b>{e(r['product'])}</b>", e(r["ranking"]), e(r["heat map"]), e(r["cost"]), e(r["generative"]),
           f"<b>{e(r['recommended'])}</b>"] for r in per_product]
    S.append(slide("Best detector for each product: PatchCore on all three",
                   "“=” is a tie within 0.0005 (for bottle the ensemble weight is 1.0 — it is PatchCore).",
                   tbl(["product", "best ranking (image AUROC)", "best heat map (pixel AUROC · AUPRO)", "lowest cost / 1,000 parts",
                        "better VAE (feature vs pixel)", "use"], pp) + f'''
<div class="kpis k3" style="margin-top:26px">{kpi("PatchCore", "overall winner", "best ranking and lowest cost on every product", "win")}
{kpi("≤ 0.002", "ensemble's only edge", "heat-map metrics on screw and carpet — a statistical tie")}
{kpi("3 / 3", "products where the feature VAE beats the pixel VAE", "0.997 / 0.798 / 0.902 vs 0.902 / 0.477 / 0.671")}</div>'''))

    # 15 ── deployed results
    mats = []
    for k in cats:
        r = rates[k]
        ng, nd = int(r["n_good"]), int(r["n_defect"])
        g = [round(ng * (1 - r["false_reject_rate"] - r["good_check_rate"])), round(ng * r["good_check_rate"]), round(ng * r["false_reject_rate"])]
        dd = [round(nd * r["escape_rate"]), round(nd * r["defect_check_rate"])]
        dd.append(nd - dd[0] - dd[1])
        cell = lambda v, good, col: (f'<td class="m {"okc" if (good and col == 0) or (not good and col == 2) else ("chk" if col == 1 else ("bad" if v else ""))}">{v}</td>')  # noqa: E731
        mats.append(f'''<div class="mat"><h3>{k} <span>{e(r["model"])}</span></h3><table class="cm"><tr><th></th><th>PASS</th><th>CHECK</th><th>REJECT</th></tr>
<tr><th>good ({ng})</th>{"".join(cell(v, True, i) for i, v in enumerate(g))}</tr>
<tr><th>defective ({nd})</th>{"".join(cell(v, False, i) for i, v in enumerate(dd))}</tr></table>
<p>false rejects <b>{100 * r["false_reject_rate"]:.0f}%</b> · escapes <b>{100 * r["escape_rate"]:.1f}%</b> · to a human <b>{100 * r["check_rate"]:.1f}%</b></p></div>''')
    strip = ""
    for k, f_ in (("bottle", "defect_broken_large_004.png"), ("screw", "defect_scratch_neck_003.png"), ("carpet", "defect_hole_004.png")):
        r_ = insp.inspect(read_rgb(c.root / "demo_images" / k / f_), k, "patchcore", compare=False, lot=insp.shipped_lot(k))
        strip += (f'<figure><img src="{th(r_["display"], 260)}" alt="{k}"><figcaption>{k} · {f_.split("_")[1]} {f_.split("_")[2] if k != "carpet" else ""}</figcaption></figure>'
                  f'<figure><img src="{th(r_["overlay"], 260)}" alt="{k} heat map"><figcaption>{dec(r_["decision"])} score {r_["score"]:.1f}</figcaption></figure>')
    S.append(slide("Deployed on held-out real defects: no good part scrapped; most defects rejected automatically",
                   "Confusion matrices of the deployed decisions (operating test split, never used for any choice).",
                   f'<div class="mats">{"".join(mats)}</div><h3>Heat maps of real held-out defects (deployed models)</h3>'
                   f'<div class="strip">{strip}</div>'))

    # 16 ── new lots + drift
    rq = d.requal.set_index("category")
    cbar = [("carpet, trained-lot model", rq.loc["carpet", "base_review_rate"] * 100, GREY, "share of parts to CHECK before re-qualification"),
            ("carpet, re-qualified", rq.loc["carpet", "requal_review_rate"] * 100, BLUE, "share of parts to CHECK after re-qualification")]
    ptab = [[k, f"{rq.loc[k, 'p_value']:.4f}", "lot change → re-qualified" if rq.loc[k, "deployed"] else "no change → keep model"]
            for k in cats]
    S.append(slide("A new carpet roll looked different: re-qualifying from 10 good parts cut human checks from 49.6% to 5.7%",
                   "Label-free: a rank test spots the lot change, new-lot patches join the bank, scale and threshold are re-fitted.", f'''
<div class="cols2"><div><h3>Share of parts sent to a human (CHECK)</h3>{hbar(cbar, 60, lambda v: f"{v:.1f}%", title="CHECK share before and after re-qualification")}
<p class="note">Price: escapes {100 * rq.loc["carpet", "base_escape_rate"]:.1f}% → {100 * rq.loc["carpet", "requal_escape_rate"]:.1f}% —
still far better than manual inspection (slide 19).</p>
<h3>Lot-shift test (Mann–Whitney U, p &lt; 0.01)</h3>{tbl(["product", "p-value", "decision"], ptab, num=(1,))}</div>
<div><h3>Camera and lighting drift</h3><ul class="tight">
<li><b>Health gate on every image:</b> brightness, contrast, sharpness z-score; |z| &gt; 4 → no automatic decision, part → CHECK.</li>
<li><b>EWMA control charts</b> on scores and camera signals (λ = 0.2, ±3σ, 2 points in a row) + <b>PSI</b> of the score histogram.</li>
<li>Camera moved → fix the camera; good parts score higher → new lot → re-qualify.</li></ul>
<h3 style="margin-top:18px">Re-qualification in four steps (no defect images)</h3>
<div class="chain"><div><b>1</b>collect ~10 good parts of the new lot</div><span>→</span><div><b>2</b>rank test: do they score
higher? (p &lt; 0.01)</div><span>→</span><div><b>3</b>add 10% of their patches to the memory bank</div><span>→</span>
<div><b>4</b>re-fit scale + threshold with synthetic defects</div></div></div></div>'''))

    # 17 ── few-shot
    g = d.few_shot.groupby("shots").image_auroc.agg(["mean", "min", "max"])
    pts = [(int(k), float(r["mean"]), float(r["min"]), float(r["max"])) for k, r in g.iterrows()]
    eight = g.loc[8, "mean"] if 8 in g.index else float("nan")
    S.append(slide(f"A new product in seconds: 8 good images give {eight:.3f} AUROC (metal nut)",
                   "PatchCore needs no network training: build the bank from a few good images, calibrate leave-one-out.",
                   f'<div class="cols2 wide-left"><div>{line_chart(pts, "Few-shot learning curve, metal nut")}</div><div>'
                   '<ul class="tight"><li>line = mean of random draws, band = range</li>'
                   '<li>1 image knows one pose and one lighting; 8 already cover most normal variation</li>'
                   '<li>the curve flattens after 16 images</li><li>live in the app: 🆕 New product → Build model</li></ul></div></div>'))

    # 18 ── one shift
    order_rows = c.rw_views
    inspectors = [(f"{p} · {('before re-qual.' if 'before' in m else 're-qualified') if p == 'carpet' else 'deployed'}",
                   v["inspectors_needed"], BLUE if "before" not in m else GREY, f"{v['inspectors_needed']:.1f} inspectors per shift")
                  for p, m, v in order_rows] + [("100% manual (status quo)", man["inspectors_needed"], "#d03b3b",
                                                 f"{man['inspectors_needed']:.0f} inspectors per shift")]
    from defectspotter.operations import shift_table
    st = shift_table(c.rw_rows, sc)
    S.append(slide(f"Per 8-hour shift: at most 1.4 inspectors instead of 25, and {c.saving_vs_manual:.0f}% lower cost on average",
                   "20,000 parts/shift, 1% defective, 36 s per human check at 30 USD/h, inspectors catch 80%, escape 400 USD.", f'''
<div class="cols2"><div><h3>Inspectors needed per shift</h3>{hbar(inspectors, 25, lambda v: f"{v:.1f}", label_w=260, width=720, title="Inspectors needed per shift")}</div>
<div>{tbl(["product", "model", "to a human", "inspectors", "escapes", "cost / shift (USD)"],
          [[r["product"], r["model"], r["parts to a human"], r["inspectors needed"], r["defects escaping"], r["cost per shift (USD)"]]
           for r in st.to_dict("records")], num=(3, 4, 5), hl=(len(st) - 1,), cls="small")}
<p class="note">Sources: BLS inspector pay (22.82 USD/h + overhead), See 2015 (≈ 80% human hit rate), 1-10-100 cost-of-quality rule.</p></div></div>'''))

    # 19 ── live demo
    S.append(slide("Try it live: one app from the story to the inspection", "bhkr0901-defectspotter.hf.space · runs on CPU with ONNX Runtime.", f'''
<div class="cols2 wide-left"><div><img class="shot1" src="{c.shot("inspect")}" alt="live demo"></div><div>
<ol class="tight"><li><b>🎬 Live demo:</b> upload <code>demo_images/screw/defect_scratch_head_009.png</code> → REJECT + heat map</li>
<li>Upload <code>good_025.png</code> → PASS; move the threshold slider</li>
<li><b>Batch</b> → CHECK parts land in the <b>review queue</b> (audit log)</li>
<li><b>Drift &amp; new lots:</b> lamp failure → CHECK; re-qualify carpet</li>
<li><b>Results → Model comparison:</b> the five detectors</li></ol>
<div class="callout">Also: Python API (<code>/inspect_part</code>, <code>/inspect_batch</code>, <code>/health</code>), Colab notebook,
complete HTML guide, 32 automated tests.</div></div></div>'''))

    # 20 ── takeaways
    S.append(slide("Takeaways", "", f'''
<div class="takeaways">
<div><b>1</b><p><b>Good parts are enough.</b> No defect image trains a model or sets a threshold — painted and diffusion
defects stand in.</p></div>
<div><b>2</b><p><b>Five detectors, one winner.</b> PatchCore on a frozen ImageNet ResNet-18: mean AUROC {pc.auroc_mean:.3f}, best on every product.</p></div>
<div><b>3</b><p><b>ResNet and VAE are separate paths.</b> Only the feature-space VAE uses ResNet features; it beats the pixel VAE
({d.macro("fvae"):.3f} vs {d.macro("vae"):.3f}).</p></div>
<div><b>4</b><p><b>Decisions priced in money.</b> PASS / CHECK / REJECT from a cost model: 0% false rejects, ≤ 1.4 inspectors per
shift, {c.saving_vs_manual:.0f}% cheaper than manual.</p></div>
<div><b>5</b><p><b>Safe in operation.</b> Camera-health gate, drift charts, human review with audit log, label-free
re-qualification for new lots.</p></div></div>''', "end"))
    return S


# --------------------------------------------------------------------------------------------------------- page
CSS = r"""
:root { --ink:#0b0b0b; --ink2:#3d3d39; --muted:#64635d; --line:#e1e0d9; --bg:#1d2024; --paper:#ffffff; --soft:#f5f5f2;
  --accent:#2a78d6; --accent-bg:#eaf2fc; --pass:#0ca30c; --pass-bg:#eaf7ea; --pass-ink:#006300; --check:#e3a008; --check-bg:#fff6dc;
  --check-ink:#7a5200; --reject:#d03b3b; --reject-bg:#fdeceb; --reject-ink:#9b1c1c;
  --d-proc:#fff; --d-proc-line:#a9a89f; --d-data:#eaf2fc; --d-data-line:#2a78d6; --d-model:#efedfb; --d-model-line:#6250d6;
  --d-gate:#fff6dc; --d-gate-line:#c98a00; --d-human:#e6f6ef; --d-human-line:#13865d; --d-alarm:#fdeceb; --d-alarm-line:#d03b3b;
  --d-store:#f3f3f1; --d-store-line:#7d7c74; --d-lane:#f7f7f5; --d-edge:#6f6e68; --surface:#fff; --s:1; color-scheme: light; }
* { box-sizing: border-box; }
html, body { margin: 0; height: 100%; background: var(--bg); font-family: system-ui, -apple-system, "Segoe UI", Roboto, Arial, sans-serif; color: var(--ink); overflow: hidden; }
.deck { position: fixed; inset: 0; }
.frame { position: absolute; inset: 0; display: none; }
.frame.on { display: block; }
.slide { position: absolute; left: 50%; top: 50%; width: 1600px; height: 900px; transform: translate(-50%, -50%) scale(var(--s));
  background: var(--paper); padding: 46px 64px 54px; overflow: hidden; display: flex; flex-direction: column; }
.slide header h2 { margin: 0; font-size: 38px; line-height: 1.15; letter-spacing: -.015em; max-width: 1400px; }
.slide header .sub { margin: 8px 0 0; font-size: 21px; color: var(--muted); }
.slide .body { flex: 1; min-height: 0; margin-top: 22px; font-size: 21px; line-height: 1.45; display: flex; flex-direction: column; gap: 14px; }
.slide::after { content: attr(data-n); position: absolute; right: 30px; bottom: 18px; font-size: 15px; color: var(--muted); }
.slide::before { content: "DefectSpotter"; position: absolute; left: 64px; bottom: 18px; font-size: 15px; color: var(--muted); font-weight: 600; }
.slide.title { background: radial-gradient(1200px 700px at 85% 10%, #eaf2fc, #fff 60%); justify-content: center; }
.tbox .eyebrow { font-size: 22px; font-weight: 700; letter-spacing: .06em; text-transform: uppercase; color: var(--accent); }
.tbox h1 { font-size: 96px; margin: 10px 0 8px; letter-spacing: -.03em; }
.tbox .lead { font-size: 32px; color: var(--ink2); margin: 0 0 34px; max-width: 1250px; }
.tbox .links { margin-top: 36px; font-size: 20px; color: var(--muted); }
.kpis { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.kpis.k3 { grid-template-columns: repeat(3, 1fr); }
.kpi { background: var(--soft); border: 1px solid var(--line); border-radius: 16px; padding: 16px 20px; }
.kpi .v { font-size: 44px; font-weight: 800; letter-spacing: -.02em; line-height: 1.1; }
.kpi .l { font-size: 19px; font-weight: 700; color: var(--ink2); margin-top: 4px; }
.kpi .s { font-size: 15.5px; color: var(--muted); margin-top: 4px; line-height: 1.35; }
.kpi.win { background: var(--pass-bg); border-color: var(--pass); }
.cols2 { display: grid; grid-template-columns: 1fr 1fr; gap: 34px; min-height: 0; }
.cols2.wide-left { grid-template-columns: 1.25fr 1fr; }
.cols2 > * { min-width: 0; }
h3 { font-size: 22px; margin: 0 0 10px; }
.card { border: 1.5px solid var(--line); border-radius: 16px; padding: 14px 20px; margin-bottom: 14px; background: #fff; }
.card .big { font-size: 28px; font-weight: 800; }
.card p { margin: 2px 0; color: var(--ink2); }
.card .price { font-size: 30px; font-weight: 800; color: var(--ink); }
.card.err { border-color: var(--reject); background: var(--reject-bg); }
.card.ok { border-color: var(--check); background: var(--check-bg); }
.big-card { height: 100%; }
.callout { background: var(--accent-bg); border-left: 5px solid var(--accent); border-radius: 10px; padding: 12px 18px; font-size: 19px; margin-top: 12px; }
.note { font-size: 17px; color: var(--muted); margin: 6px 0; }
ul.tight, ol.tight { margin: 0; padding-left: 26px; } ul.tight li, ol.tight li { margin: 6px 0; }
code { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: .85em; background: var(--soft); border: 1px solid var(--line); border-radius: 5px; padding: 0 5px; overflow-wrap: anywhere; }
table.t { border-collapse: collapse; width: 100%; font-size: 18px; }
table.t.small { font-size: 16.5px; }
table.t th { text-align: left; font-weight: 700; color: var(--ink2); border-bottom: 2px solid var(--line); padding: 7px 10px; background: var(--soft); }
table.t td { border-bottom: 1px solid var(--line); padding: 7px 10px; vertical-align: top; }
table.t .num { text-align: right; font-variant-numeric: tabular-nums; }
table.t tr.hl td { background: var(--pass-bg); }
.sw { display: inline-block; width: 12px; height: 12px; border-radius: 3px; margin-right: 8px; vertical-align: 0; }
.dec { display: inline-block; font-weight: 800; font-size: .9em; padding: 1px 10px; border-radius: 999px; border: 1.5px solid; }
.dec.PASS { color: var(--pass-ink); background: var(--pass-bg); border-color: var(--pass); }
.dec.CHECK { color: var(--check-ink); background: var(--check-bg); border-color: var(--check); }
.dec.REJECT { color: var(--reject-ink); background: var(--reject-bg); border-color: var(--reject); }
.zrow { display: grid; grid-template-columns: 170px 1fr; gap: 12px; align-items: center; margin: 8px 0; }
.zname { font-weight: 700; font-size: 18px; }
.zones { display: grid; border-radius: 10px; overflow: hidden; font-size: 16px; font-weight: 700; border: 1px solid var(--line); }
.zones div { padding: 8px 10px; white-space: nowrap; overflow: hidden; }
.zones .p { background: var(--pass-bg); color: var(--pass-ink); }
.zones .c { background: var(--check-bg); color: var(--check-ink); text-align: center; border-left: 2px solid var(--check); border-right: 2px solid var(--check); padding: 8px 0 !important; font-size: 13px; }
.zones .r { background: var(--reject-bg); color: var(--reject-ink); text-align: right; }
.flow { flex: 1; min-height: 0; display: flex; align-items: center; justify-content: center; }
.flow svg { max-width: 100%; max-height: 100%; width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 18px; font-size: 15px; color: var(--ink2); }
.legend i { display: inline-block; width: 15px; height: 12px; border-radius: 3px; border: 1.5px solid; margin-right: 6px; vertical-align: -1px; }
svg .n-proc { fill: var(--d-proc); stroke: var(--d-proc-line); } svg .n-data { fill: var(--d-data); stroke: var(--d-data-line); }
svg .n-model { fill: var(--d-model); stroke: var(--d-model-line); } svg .n-gate { fill: var(--d-gate); stroke: var(--d-gate-line); }
svg .n-human { fill: var(--d-human); stroke: var(--d-human-line); } svg .n-alarm { fill: var(--d-alarm); stroke: var(--d-alarm-line); }
svg .n-store { fill: var(--d-store); stroke: var(--d-store-line); } svg .n-ghost { fill: transparent; stroke: var(--d-proc-line); stroke-dasharray: 5 4; }
svg .n-best { fill: var(--pass-bg); stroke: var(--pass); stroke-width: 2.4; }
svg .shape { stroke-width: 1.6; } svg .t1 { fill: var(--ink); font: 600 14px system-ui, sans-serif; } svg .t2 { fill: var(--ink2); font: 12.5px system-ui, sans-serif; }
svg .mono { font-family: ui-monospace, Menlo, Consolas, monospace; } svg .edge { fill: none; stroke: var(--d-edge); stroke-width: 1.7; }
svg .edge.dash { stroke-dasharray: 6 5; } svg .ah { fill: var(--d-edge); }
svg .el { fill: var(--ink2); font: 600 12px system-ui, sans-serif; paint-order: stroke; stroke: #fff; stroke-width: 5px; stroke-linejoin: round; }
svg .lane { fill: var(--d-lane); stroke: var(--line); } svg .lane-t { fill: var(--muted); font: 700 12.5px system-ui, sans-serif; }
svg .ann { fill: var(--muted); font: italic 12px system-ui, sans-serif; }
svg.chart { width: 100%; height: auto; } svg.chart .grid { stroke: #ecebe6; } svg.chart .tick { fill: var(--muted); font-size: 14px; }
svg.chart .lab { fill: var(--ink); font-size: 16px; font-weight: 600; } svg.chart .val { fill: var(--ink); font-size: 16px; font-weight: 700; }
svg.chart .axis { fill: var(--ink2); font-size: 15px; } svg.chart rect:hover, svg.chart circle:hover { opacity: .8; }
.facts { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; font-size: 17px; }
.facts div { background: var(--soft); border: 1px solid var(--line); border-radius: 12px; padding: 10px 14px; line-height: 1.35; }
.split { display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 18px; font-size: 16px; }
.split div { border-radius: 12px; padding: 10px 12px; border: 1.5px solid; }
.split span { display: block; color: var(--ink2); font-size: 14.5px; margin-top: 3px; }
.split .r1 { background: var(--d-model); border-color: var(--d-model-line); } .split .r2 { background: #fff; border-color: var(--d-proc-line); }
.split .r3 { border-style: dashed; border-color: var(--d-proc-line); } .split .r4 { background: var(--d-gate); border-color: var(--d-gate-line); }
.imgs1 { display: flex; flex-direction: column; gap: 10px; min-height: 0; }
.imgs1 img { width: 100%; max-height: 196px; object-fit: contain; border-radius: 10px; border: 1px solid var(--line); background: #000; }
.thumbs { display: grid; grid-template-columns: repeat(6, 196px); gap: 14px; justify-content: start; }
.thumbs figure { margin: 0; } .thumbs img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 10px; border: 1px solid var(--line); }
.thumbs figcaption { font-size: 14.5px; color: var(--muted); text-align: center; margin-top: 3px; }
.shot { margin-bottom: 16px; } .pair { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }
.pair img { width: 100%; border-radius: 12px; border: 1px solid var(--line); }
.shot .cap { font-size: 18px; margin-top: 6px; }
.pairs { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.pp { display: grid; grid-template-columns: 1fr 26px 1fr; align-items: center; gap: 6px; }
.pp img { width: 100%; border-radius: 10px; border: 1px solid var(--line); } .pp span { text-align: center; color: var(--muted); font-size: 22px; }
.mini { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 16px; background: var(--soft); border-radius: 10px; padding: 10px 12px; margin: 6px 0 12px; }
.formula { font-family: ui-monospace, Menlo, Consolas, monospace; font-size: 20px; background: var(--soft); border: 1px solid var(--line); border-radius: 12px; padding: 14px 18px; margin-bottom: 14px; }
.formula span { display: block; font-family: system-ui, sans-serif; font-size: 16px; color: var(--muted); margin-top: 6px; }
.mats { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; }
.mat h3 span { font-weight: 500; color: var(--muted); font-size: 16px; }
.mat p { font-size: 16.5px; margin: 8px 0 0; }
table.cm { border-collapse: separate; border-spacing: 4px; width: 100%; font-size: 17px; }
table.cm th { font-weight: 700; color: var(--ink2); text-align: center; font-size: 15px; }
table.cm td.m { text-align: center; font-size: 26px; font-weight: 800; border-radius: 10px; padding: 10px 0; background: var(--soft); }
table.cm td.okc { background: var(--pass-bg); color: var(--pass-ink); } table.cm td.chk { background: var(--check-bg); color: var(--check-ink); }
table.cm td.bad { background: var(--reject-bg); color: var(--reject-ink); }
img.wide { width: 100%; max-height: 330px; object-fit: contain; border-radius: 12px; background: #000; }
img.shot1 { width: 100%; max-height: 640px; object-fit: contain; object-position: top; border-radius: 12px; border: 1px solid var(--line); box-shadow: 0 6px 24px rgba(0,0,0,.12); }
.takeaways { display: grid; grid-template-columns: 1fr; gap: 16px; }
.takeaways div { display: grid; grid-template-columns: 64px 1fr; gap: 16px; align-items: center; background: var(--soft); border: 1px solid var(--line); border-radius: 16px; padding: 14px 20px; }
.takeaways div > b { width: 52px; height: 52px; border-radius: 50%; background: var(--accent); color: #fff; display: grid; place-items: center; font-size: 26px; }
.takeaways p { margin: 0; font-size: 23px; }
.dcards { display: grid; grid-template-columns: repeat(4, 1fr); gap: 18px; flex: 1; min-height: 0; }
.dcard { border: 1.5px solid var(--line); border-radius: 18px; padding: 16px 18px; display: flex; flex-direction: column; gap: 10px; background: #fff; }
.dcard.win { border: 3px solid var(--pass); background: var(--pass-bg); }
.dhead { display: flex; align-items: center; gap: 10px; font-size: 24px; border-bottom: 5px solid; padding-bottom: 8px; }
.dhead .num { font-size: 28px; }
.uses { font-weight: 800; font-size: 17px; border-radius: 10px; padding: 7px 10px; }
.uses.yes { background: var(--d-model); color: #3f2fb3; border: 1.5px solid var(--d-model-line); }
.uses.no { background: var(--soft); color: var(--ink2); border: 1.5px dashed var(--d-proc-line); }
.dcard .row { font-size: 17px; line-height: 1.35; } .dcard .row span { display: block; font-size: 13.5px; font-weight: 700; text-transform: uppercase; letter-spacing: .06em; color: var(--muted); }
.auc .v { font-size: 52px; font-weight: 800; letter-spacing: -.02em; line-height: 1; }
.track { height: 14px; border-radius: 7px; background: #ecebe6; margin: 8px 0 6px; overflow: hidden; }
.fill { height: 100%; border-radius: 7px; }
.per { font-size: 15px; color: var(--muted); }
.dcard .foot { margin-top: auto; display: flex; justify-content: space-between; align-items: center; font-size: 15.5px; color: var(--muted); }
.vd { font-weight: 800; color: var(--ink); background: var(--soft); border-radius: 999px; padding: 3px 12px; border: 1px solid var(--line); }
.dcard.win .vd { background: var(--pass); color: #fff; border-color: var(--pass); }
.four { display: grid; grid-template-columns: repeat(4, 1fr); gap: 16px; }
.four figure, .strip figure { margin: 0; } .four img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 12px; border: 1px solid var(--line); }
.four figcaption, .strip figcaption { font-size: 17px; margin-top: 6px; }
.strip { display: grid; grid-template-columns: repeat(6, 1fr); gap: 12px; }
.strip img { width: 100%; aspect-ratio: 1; object-fit: cover; border-radius: 10px; border: 1px solid var(--line); }
.flow.fixed { flex: none; height: 330px; }
.chain { display: grid; grid-template-columns: 1fr 22px 1fr 22px 1fr 22px 1fr; align-items: stretch; gap: 6px; font-size: 16px; }
.chain div { background: var(--d-model); border: 1.5px solid var(--d-model-line); border-radius: 12px; padding: 10px 10px; line-height: 1.3; }
.chain div b { display: block; font-size: 22px; color: #3f2fb3; }
.chain span { align-self: center; text-align: center; color: var(--muted); font-size: 22px; }
/* chrome */
.bar { position: fixed; left: 0; bottom: 0; height: 4px; background: var(--accent); transition: width .2s; z-index: 5; }
.nav { position: fixed; right: 14px; top: 12px; z-index: 6; display: flex; gap: 6px; opacity: .25; transition: opacity .2s; }
.nav:hover { opacity: 1; }
.nav button { font: 600 14px system-ui; padding: 6px 12px; border-radius: 8px; border: 1px solid #555; background: #2a2d33; color: #fff; cursor: pointer; }
.hint { position: fixed; left: 50%; bottom: 14px; transform: translateX(-50%); color: #cfcfcf; font: 14px system-ui; z-index: 6; transition: opacity .6s; }
body.overview { overflow: auto; }
body.overview .deck { position: static; display: grid; grid-template-columns: repeat(auto-fill, 352px); gap: 16px; padding: 20px; justify-content: center; }
body.overview .frame { display: block; position: relative; width: 352px; height: 198px; overflow: hidden; cursor: pointer; border-radius: 8px; outline: 2px solid transparent; }
body.overview .frame:hover { outline-color: var(--accent); }
body.overview .slide { left: 0; top: 0; transform: scale(.22); transform-origin: top left; }
body.overview .nav, body.overview .hint, body.overview .bar { display: none; }
@media print {
  @page { size: 1600px 900px; margin: 0; }
  html, body { overflow: visible; height: auto; background: #fff; }
  .deck { position: static; } .frame { display: block !important; position: relative; width: 1600px; height: 900px; page-break-after: always; break-after: page; }
  .slide { transform: none; left: 0; top: 0; } .nav, .hint, .bar { display: none; }
}
"""

JS = r"""
(function () {
  var frames, i = 0;
  function fit() { var s = Math.min(window.innerWidth / 1600, window.innerHeight / 900); document.documentElement.style.setProperty('--s', s); }
  function show(k) {
    i = Math.max(0, Math.min(frames.length - 1, k));
    frames.forEach(function (f, j) { f.classList.toggle('on', j === i); });
    document.querySelector('.bar').style.width = (100 * (i + 1) / frames.length) + '%';
    try { history.replaceState(null, '', '#' + (i + 1)); } catch (e) {}
  }
  function overview(on) { document.body.classList.toggle('overview', on); if (!on) show(i); }
  document.addEventListener('DOMContentLoaded', function () {
    frames = Array.prototype.slice.call(document.querySelectorAll('.frame'));
    frames.forEach(function (f, j) {
      f.querySelector('.slide').setAttribute('data-n', (j + 1) + ' / ' + frames.length);
      f.addEventListener('click', function () { if (document.body.classList.contains('overview')) { i = j; overview(false); } });
    });
    var h = parseInt((location.hash || '#1').slice(1), 10); show(isNaN(h) ? 0 : h - 1); fit();
    window.addEventListener('resize', fit);
    document.getElementById('prev').onclick = function () { show(i - 1); };
    document.getElementById('next').onclick = function () { show(i + 1); };
    document.getElementById('ov').onclick = function () { overview(!document.body.classList.contains('overview')); };
    document.getElementById('fs').onclick = function () { if (!document.fullscreenElement) document.documentElement.requestFullscreen(); else document.exitFullscreen(); };
    setTimeout(function () { var t = document.querySelector('.hint'); if (t) t.style.opacity = 0; }, 6000);
    var x0 = null;
    document.addEventListener('touchstart', function (ev) { x0 = ev.touches[0].clientX; }, { passive: true });
    document.addEventListener('touchend', function (ev) { if (x0 === null) return; var dx = ev.changedTouches[0].clientX - x0; if (Math.abs(dx) > 50) show(i + (dx < 0 ? 1 : -1)); x0 = null; });
  });
  document.addEventListener('keydown', function (ev) {
    if (document.body.classList.contains('overview')) { if (ev.key === 'o' || ev.key === 'O' || ev.key === 'Escape') overview(false); return; }
    var k = ev.key;
    if (k === 'ArrowRight' || k === 'PageDown' || k === ' ' || k === 'Enter') { show(i + 1); ev.preventDefault(); }
    else if (k === 'ArrowLeft' || k === 'PageUp' || k === 'Backspace') { show(i - 1); ev.preventDefault(); }
    else if (k === 'Home') show(0); else if (k === 'End') show(frames.length - 1);
    else if (k === 'f' || k === 'F') document.getElementById('fs').click();
    else if (k === 'o' || k === 'O') overview(true);
  });
})();
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(ROOT / "docs" / "DefectSpotter_Presentation.html"))
    args = parser.parse_args()
    c = context(ROOT)
    slides = build(c)
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DefectSpotter Presentation</title><style>{CSS}</style><script>{JS}</script></head>
<body><div class="deck">{"".join(slides)}</div>
<div class="bar"></div>
<div class="nav"><button id="prev" type="button" aria-label="previous slide">←</button><button id="next" type="button" aria-label="next slide">→</button>
<button id="ov" type="button">Overview (O)</button><button id="fs" type="button">Fullscreen (F)</button></div>
<div class="hint">← → or space to navigate · F fullscreen · O overview · print = one slide per page</div>
</body></html>"""
    out = Path(args.out)
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size / 1e6:.2f} MB, {len(slides)} slides)")


if __name__ == "__main__":
    main()
