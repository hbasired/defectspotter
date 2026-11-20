"""Inline-SVG flow diagrams for the HTML guide.

A tiny layout DSL (boxes at fixed coordinates, orthogonal edges with rounded corners); every colour comes from CSS
classes, so the diagrams follow the page's light/dark theme. ``Flow.warnings`` lists text that may overflow its box.
"""
from __future__ import annotations

import itertools
import math
from html import escape

_IDS = itertools.count(1)
KIND_LABELS = {"data": "data / images", "model": "model or learned component", "proc": "processing step",
               "gate": "decision / check", "human": "human", "store": "stored files", "alarm": "alarm / action",
               "ghost": "comparison only"}


def _e(text) -> str:
    return escape(str(text), quote=True)


def _side(side):
    return side if isinstance(side, tuple) else (side, 0.5)


def _path(points, radius: float = 9.0) -> str:
    pts = [points[0]]
    for p in points[1:]:
        if abs(p[0] - pts[-1][0]) > 0.01 or abs(p[1] - pts[-1][1]) > 0.01:
            pts.append(p)
    d = f"M{pts[0][0]:.1f},{pts[0][1]:.1f}"
    for i in range(1, len(pts) - 1):
        (x0, y0), (x1, y1), (x2, y2) = pts[i - 1], pts[i], pts[i + 1]
        d1, d2 = math.hypot(x1 - x0, y1 - y0), math.hypot(x2 - x1, y2 - y1)
        r = min(radius, d1 / 2, d2 / 2)
        ax, ay = x1 - (x1 - x0) / d1 * r, y1 - (y1 - y0) / d1 * r
        bx, by = x1 + (x2 - x1) / d2 * r, y1 + (y2 - y1) / d2 * r
        d += f" L{ax:.1f},{ay:.1f} Q{x1:.1f},{y1:.1f} {bx:.1f},{by:.1f}"
    return d + f" L{pts[-1][0]:.1f},{pts[-1][1]:.1f}", pts


class Flow:
    def __init__(self, width: int, height: int, title: str, desc: str = "") -> None:
        self.w, self.h, self.title, self.desc = width, height, title, desc
        self.uid = f"fl{next(_IDS)}"
        self.nodes: dict[str, dict] = {}
        self.edges: list[dict] = []
        self.lanes: list[tuple] = []
        self.notes: list[tuple] = []
        self.warnings: list[str] = []

    # ------------------------------------------------------------------ building
    def node(self, nid, x, y, w, h, title, sub=(), kind="proc", mono=False):
        sub = [sub] if isinstance(sub, str) else list(sub)
        self.nodes[nid] = dict(x=x, y=y, w=w, h=h, title=title, sub=sub, kind=kind, mono=mono)
        limit = w - 12
        if kind != "pt":
            if len(title) * (8.6 if mono else 7.7) > limit:
                self.warnings.append(f"[{self.title}] title '{title}' ~{len(title) * 7.7:.0f}px > {limit}px")
            for s in sub:
                if len(s) * 6.45 > limit:
                    self.warnings.append(f"[{self.title}] '{s}' ~{len(s) * 6.45:.0f}px > {limit}px")
        return self

    def point(self, nid, x, y):
        return self.node(nid, x, y, 0, 0, "", kind="pt")

    def edge(self, a, b, label="", sa="r", sb="l", via=None, dash=False, lxy=None, lpos=0.5):
        self.edges.append(dict(a=a, b=b, label=label, sa=sa, sb=sb, via=via or [], dash=dash, lxy=lxy, lpos=lpos))
        return self

    def lane(self, x, y, w, h, label):
        self.lanes.append((x, y, w, h, label))
        return self

    def note(self, x, y, text, cls="ann", anchor="start"):
        self.notes.append((x, y, text, cls, anchor))
        return self

    # ------------------------------------------------------------------ geometry
    def _anchor(self, nid, side):
        n = self.nodes[nid]
        s, f = _side(side)
        x, y, w, h = n["x"], n["y"], n["w"], n["h"]
        return {"r": (x + w, y + h * f), "l": (x, y + h * f), "t": (x + w * f, y), "b": (x + w * f, y + h)}[s]

    def _points(self, e):
        p1, p2 = self._anchor(e["a"], e["sa"]), self._anchor(e["b"], e["sb"])
        sa, sb = _side(e["sa"])[0], _side(e["sb"])[0]
        pts = [p1]
        if e["via"]:
            for vx, vy in e["via"]:
                last = pts[-1]
                pts.append((last[0] if vx is None else vx, last[1] if vy is None else vy))
            last = pts[-1]
            if abs(last[0] - p2[0]) > 0.5 and abs(last[1] - p2[1]) > 0.5:
                pts.append((p2[0], last[1]) if sb in "tb" else (last[0], p2[1]))
        else:
            (x1, y1), (x2, y2) = p1, p2
            if sa in "rl" and sb in "rl":
                if abs(y1 - y2) > 0.5:
                    mx = (x1 + x2) / 2
                    pts += [(mx, y1), (mx, y2)]
            elif sa in "tb" and sb in "tb":
                if abs(x1 - x2) > 0.5:
                    my = (y1 + y2) / 2
                    pts += [(x1, my), (x2, my)]
            elif sa in "rl":
                pts.append((x2, y1))
            else:
                pts.append((x1, y2))
        pts.append(p2)
        return pts

    # ------------------------------------------------------------------ rendering
    def _node_svg(self, n) -> str:
        x, y, w, h, kind = n["x"], n["y"], n["w"], n["h"], n["kind"]
        cls = f"shape n-{kind}"
        if kind == "gate":
            c = min(14.0, h / 2)
            pts = " ".join(f"{px:.1f},{py:.1f}" for px, py in
                           [(x + c, y), (x + w - c, y), (x + w, y + h / 2), (x + w - c, y + h), (x + c, y + h), (x, y + h / 2)])
            shape = f'<polygon class="{cls}" points="{pts}"/>'
            shift = 0.0
        elif kind == "store":
            e = 7.0
            shape = (f'<path class="{cls}" d="M{x},{y + e} V{y + h - e} A{w / 2},{e} 0 0 0 {x + w},{y + h - e} V{y + e} Z"/>'
                     f'<ellipse class="{cls}" cx="{x + w / 2}" cy="{y + e}" rx="{w / 2}" ry="{e}"/>')
            shift = e / 2
        else:
            shape = f'<rect class="{cls}" x="{x}" y="{y}" width="{w}" height="{h}" rx="10"/>'
            shift = 0.0
        lines = ([(n["title"], "t1" + (" mono" if n["mono"] else ""), 17.0, 13.0)] if n["title"] else []) \
            + [(s, "t2", 15.5, 11.5) for s in n["sub"]]
        cur = y + shift + (h - shift) / 2 - sum(line[2] for line in lines) / 2
        out = [shape]
        for text, c, line_height, baseline in lines:
            out.append(f'<text class="{c}" x="{x + w / 2:.1f}" y="{cur + baseline:.1f}" text-anchor="middle">{_e(text)}</text>')
            cur += line_height
        return "".join(out)

    def svg(self) -> str:
        u = self.uid
        out = [f'<svg viewBox="0 0 {self.w} {self.h}" role="img" aria-labelledby="{u}-t {u}-d" '
               f'xmlns="http://www.w3.org/2000/svg">',
               f'<title id="{u}-t">{_e(self.title)}</title><desc id="{u}-d">{_e(self.desc or self.title)}</desc>',
               f'<defs><marker id="{u}-ah" viewBox="0 0 10 10" refX="9.5" refY="5" markerUnits="userSpaceOnUse" '
               f'markerWidth="10" markerHeight="10" orient="auto-start-reverse"><path class="ah" d="M0,0 L10,5 L0,10 z"/>'
               f'</marker></defs>']
        for x, y, w, h, label in self.lanes:
            out.append(f'<rect class="lane" x="{x}" y="{y}" width="{w}" height="{h}" rx="14"/>')
            out.append(f'<text class="lane-t" x="{x + 16}" y="{y + 24}">{_e(label)}</text>')
        labels = []
        for e in self.edges:
            d, pts = _path(self._points(e))
            out.append(f'<path class="edge{" dash" if e["dash"] else ""}" d="{d}" marker-end="url(#{u}-ah)"/>')
            if e["label"]:
                if e["lxy"]:
                    lx, ly, anchor = e["lxy"]
                else:
                    segs = [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]
                    (ax, ay), (bx, by) = max(segs, key=lambda s: math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]))
                    mx, my = ax + (bx - ax) * e["lpos"], ay + (by - ay) * e["lpos"]
                    if abs(by - ay) < 0.5:
                        lx, ly, anchor = mx, my - 7, "middle"
                    else:
                        lx, ly, anchor = mx + 7, my + 4, "start"
                labels.append(f'<text class="el" x="{lx:.1f}" y="{ly:.1f}" text-anchor="{anchor}">{_e(e["label"])}</text>')
        for n in self.nodes.values():
            if n["kind"] != "pt":
                out.append(self._node_svg(n))
        out += labels
        for x, y, text, cls, anchor in self.notes:
            out.append(f'<text class="{cls}" x="{x}" y="{y}" text-anchor="{anchor}">{_e(text)}</text>')
        out.append("</svg>")
        return "".join(out)


def legend(kinds) -> str:
    var = {"data": "data", "model": "model", "proc": "proc", "gate": "gate", "human": "human", "store": "store",
           "alarm": "alarm", "ghost": "proc"}
    items = []
    for k in kinds:
        style = (f"background:var(--d-{var[k]});border-color:var(--d-{var[k]}-line)"
                 + (";border-style:dashed;background:transparent" if k == "ghost" else ""))
        items.append(f'<span><i style="{style}"></i>{_e(KIND_LABELS[k])}</span>')
    return '<div class="legend">' + "".join(items) + "</div>"


def figure(flow: Flow, caption: str, kinds=("data", "model", "proc", "gate")) -> str:
    return f'<div class="diagram">{flow.svg()}{legend(kinds)}<p class="cap">{caption}</p></div>'


# =====================================================================================================================
# The diagrams. ``n`` is a dict of numbers read from the reports (see build_guide.py).
# =====================================================================================================================
def system_overview(n) -> Flow:
    f = Flow(1160, 650, "DefectSpotter end to end",
             "Build once per product offline, inspect every part online, operate and re-qualify after deployment.")
    f.lane(10, 10, 1140, 244, "1 · Build — offline, once per product (scripts/train.py)")
    f.lane(10, 276, 1140, 196, "2 · Inspect — online, every part (app, API)")
    f.lane(10, 494, 1140, 146, "3 · Operate — after deployment")
    xs = [26, 216, 406, 596, 786, 976]
    a = [("a1", "Good parts only", ["MVTec AD training set,", "no defect images"], "data"),
         ("a2", "Frozen ResNet-18", ["ImageNet weights;", "not fine-tuned, no head"], "model"),
         ("a3", "Detectors", ["PatchCore memory bank,", "feature VAE, pixel VAE"], "model"),
         ("a4", "Score held-out", ["good parts never seen", "+ synthetic defects"], "proc"),
         ("a5", "Calibrate", ["0–100 score scale,", "cost-optimal threshold"], "proc"),
         ("a6", "artifacts/", [f"{n['artifact_mb']:.0f} MB: ONNX, banks,", "calibration.json"], "store")]
    for x, (nid, t, s, k) in zip(xs, a):
        f.node(nid, x, 50, 160, 92, t, s, k)
    f.node("s1", 596, 176, 160, 64, "Synthetic defects", ["painted + Stable", "Diffusion, no labels"], "data")
    for p, q in zip("a1 a2 a3 a4 a5".split(), "a2 a3 a4 a5 a6".split()):
        f.edge(p, q)
    f.edge("s1", "a4", sa="t", sb="b")
    b = [("b1", "Part image", ["camera, upload or API;", "any size, RGB"], "data"),
         ("b2", "Health gate", ["brightness, contrast,", "sharpness within ±4σ?"], "gate"),
         ("b3", "Backbone features", ["pose-normalised screws;", "384 numbers per patch"], "model"),
         ("b4", "Nearest neighbour", ["distance of every patch", "to the memory bank"], "model"),
         ("b5", "0–100 score + map", ["25 = typical good part,", "75 = typical defect"], "proc"),
         ("b6", "Decision", ["PASS · CHECK · REJECT", "+ heat map + regions"], "gate")]
    for x, (nid, t, s, k) in zip(xs, b):
        f.node(nid, x, 318, 160, 92, t, s, k)
    for p, q in zip("b1 b2 b3 b4 b5".split(), "b2 b3 b4 b5 b6".split()):
        f.edge(p, q)
    f.edge("a6", "b4", sa="b", sb="t", via=[(None, 265), (676, None)], lxy=(1046, 212, "end"), label="loaded by the app")
    c = [("c4", 411, "Re-qualification", ["new-lot patches join", "the bank; re-fit t"], "proc"),
         ("c3", 601, "Lot changed?", ["Mann–Whitney U", "on ~10 good parts"], "gate"),
         ("c2", 791, "Drift monitoring", ["EWMA charts on", "scores + camera"], "proc"),
         ("c1", 981, "Review queue", ["inspector decides", "each CHECK; audit"], "human")]
    for nid, x, t, s, k in c:
        f.node(nid, x, 534, 150, 88, t, s, k)
    f.edge("b6", "c1", sa="b", sb="t", label="CHECK")
    f.edge("b5", "c2", sa="b", sb="t", label="scores")
    f.edge("c2", "c3", sa="l", sb="r", label="alarm")
    f.edge("c3", "c4", sa="l", sb="r", label="yes")
    f.edge("c4", "b4", sa=("t", 0.5), sb=("b", 0.3), via=[(None, 483), (644, None)], dash=True,
           label="updated bank + threshold")
    f.note(26, 566, "Nothing is retrained silently:")
    f.note(26, 584, "every change is a recorded,")
    f.note(26, 602, "re-validated release.")
    return f


def build_pipeline(n) -> Flow:
    f = Flow(1160, 420, "The build pipeline", "Seventeen resumable stages run by scripts/train.py, grouped in seven phases.")
    cols = [("1 · Data", [("index", "leakage-free splits"), ("eda", "statistics, montages")]),
            ("2 · Defects", [("synthetic", "3 painted kinds"), ("SD images", "Stable Diffusion")]),
            ("3 · Detectors", [("patchcore", "recipe + bank"), ("vae", "pixel VAE, 120 ep."),
                               ("fvae", "feature VAE, 60 ep."), ("mean", "mean ± std image")]),
            ("4 · Stress", [("severity", "50% / 25% replays"), ("diffusion", "score SD defects")]),
            ("5 · Calibrate", [("calibrate", "scales + thresholds")]),
            ("6 · Evaluate", [("evaluate", "test-split metrics"), ("requalify", "lot-shift test"),
                              ("latency", "ms per part"), ("drift", "stress + charts"),
                              ("few_shot", "new product")]),
            ("7 · Package", [("examples", "app images"), ("summary", "summary.json")])]
    x = 12
    for i, (title, stages) in enumerate(cols):
        f.lane(x, 40, 152, 370, title)
        y = 80
        for name, sub in stages:
            f.node(f"s{i}{name}", x + 8, y, 136, 54, name, [sub], "proc" if name != "SD images" else "ghost",
                   mono=name != "SD images")
            y += 64
        x += 164
    f.node("art", 996 + 8, 216, 136, 70, "artifacts/", [f"{n['artifact_mb']:.0f} MB of models"], "store")
    f.node("rep", 996 + 8, 300, 136, 70, "reports/", ["CSV, JSON, figures"], "store")
    f.point("t0", 14, 18)
    f.point("t1", 1146, 18)
    f.edge("t0", "t1", label="stages run left to right · each caches its results · any stage can be re-run alone",
           sa="r", sb="l", lxy=(580, 14, "middle"))
    return f


def inspection_flow(n) -> Flow:
    f = Flow(1160, 330, "Inspecting one part", "The ten steps between an image and a decision.")
    top = [("i1", "1 · Part image", ["any resolution, RGB;", "upload, camera, API"], "data"),
           ("i2", "2 · Health check", ["brightness, contrast and", "sharpness z-scores"], "gate"),
           ("i3", "3 · Pose (screws)", ["PCA axis of the part mask;", "rotate head-left, centre"], "proc"),
           ("i4", "4 · Resize + normalise", ["224 px (bottle, carpet),", "288 px (screw); ImageNet"], "proc"),
           ("i5", "5 · ResNet-18 (ONNX)", ["layer2 + layer3 → 384-d", "per patch, 28×28 / 36×36"], "model")]
    bottom = [("i6", "6 · Nearest neighbour", ["distance to the closest of", f"{n['bank_range']} bank vectors"], "model"),
              ("i7", "7 · Anomaly map", ["upsample, Gaussian blur,", "warp back to the image"], "proc"),
              ("i8", "8 · Score", ["max or top-k mean →", "0–100 scale (25 / 75)"], "proc"),
              ("i9", "9 · Decision", ["PASS · CHECK · REJECT", "from threshold ± band"], "gate"),
              ("i10", "10 · Result", ["heat map, regions, ‘why’,", "JSON, review queue"], "data")]
    xs = [20, 250, 480, 710, 940]
    for x, (nid, t, s, k) in zip(xs, top):
        f.node(nid, x, 34, 200, 96, t, s, k)
    for x, (nid, t, s, k) in zip(xs[::-1], bottom):
        f.node(nid, x, 208, 200, 96, t, s, k)
    for p, q in zip("i1 i2 i3 i4".split(), "i2 i3 i4 i5".split()):
        f.edge(p, q)
    f.edge("i5", "i6", sa="b", sb="t")
    for p, q in zip("i6 i7 i8 i9".split(), "i7 i8 i9 i10".split()):
        f.edge(p, q, sa="l", sb="r")
    f.edge("i2", "i9", sa="b", sb="t", dash=True, label="|z| > 4 → CHECK", lxy=(358, 174, "start"))
    return f


def patchcore_flow(n) -> Flow:
    f = Flow(1160, 340, "PatchCore: build and query", "How the memory bank is built and how a new part is scored.")
    f.lane(10, 8, 1140, 150, "Build — once, no gradient descent")
    f.lane(10, 178, 1140, 154, "Query — every part")
    xs = [26, 252, 478, 704, 930]
    build = [("p1", "Good images", [n["train_counts"], "(model split)"], "data"),
             ("p2", "Backbone", ["784 or 1,296 patches per", "image, 384 numbers each"], "model"),
             ("p3", "All patch vectors", [n["patch_counts"], "vectors per product"], "data"),
             ("p4", "Greedy coreset", ["keep 1%: always add the", "patch farthest from kept"], "proc"),
             ("p5", "Memory bank", [n["bank_counts"], "vectors, float16"], "store")]
    query = [("q1", "New part", ["same backbone, same", "resolution and pose"], "data"),
             ("q2", "Patch vectors", ["28×28 or 36×36 grid of", "384-d descriptors"], "model"),
             ("q3", "Nearest neighbour", ["Euclidean distance to the", "closest bank vector"], "proc"),
             ("q4", "Distance map", ["upsample to the image,", "Gaussian σ = 2–4 px"], "proc"),
             ("q5", "Image score", ["largest patch distance", "or top-k mean of map"], "gate")]
    for x, (nid, t, s, k) in zip(xs, build):
        f.node(nid, x, 46, 196, 92, t, s, k)
    for x, (nid, t, s, k) in zip(xs, query):
        f.node(nid, x, 220, 196, 92, t, s, k)
    for row in (["p1", "p2", "p3", "p4", "p5"], ["q1", "q2", "q3", "q4", "q5"]):
        for p, q in zip(row, row[1:]):
            f.edge(p, q)
    f.edge("p5", "q3", sa="b", sb="t", via=[(None, 168), (576, None)], label="searched for every patch",
           lxy=(800, 192, "middle"))
    return f


def vae_flow(n) -> Flow:
    f = Flow(1160, 372, "The two variational autoencoders",
             "Pixel VAE rebuilds the image; the feature-space VAE rebuilds the backbone's patch descriptors.")
    f.lane(10, 8, 1140, 170, "Pixel VAE (v1) — rebuilds the image")
    f.lane(10, 194, 1140, 170, "Feature-space VAE (v3) — rebuilds what the backbone sees")
    xs = [26, 186, 346, 506, 666, 826, 986]
    pix = [("v1", "Image", ["3 × 128 × 128", "values 0–1"], "data"),
           ("v2", "Encoder", ["4 × stride-2 conv", "32→64→128→128 ch"], "model"),
           ("v3", "μ, log σ²", ["128 numbers each", "(fully connected)"], "model"),
           ("v4", "Sample z", ["z = μ + σ·ε,", "ε ~ N(0, 1)"], "proc"),
           ("v5", "Decoder", ["mirror: transposed", "convs, sigmoid"], "model"),
           ("v6", "Reconstruction", ["3 × 128 × 128,", "a ‘good’ version"], "data"),
           ("v7", "Error map", ["0.5·|x − x̂| +", "0.5·(1 − SSIM)"], "gate")]
    feat = [("f1", "Features", ["384 × 28 × 28,", "normalised"], "data"),
            ("f2", "Encoder", ["1×1 conv 384→256,", "2 × 3×3 stride 2"], "model"),
            ("f3", "μ, log σ²", ["32 × 7 × 7", "spatial latent"], "model"),
            ("f4", "Sample z", ["z = μ + σ·ε", "(μ at test time)"], "proc"),
            ("f5", "Decoder", ["2 × 4×4 transposed", "conv, 1×1 → 384"], "model"),
            ("f6", "Reconstruction", ["384 × 28 × 28,", "normal features"], "data"),
            ("f7", "Error map", ["per-patch squared", "error → upsample"], "gate")]
    for x, (nid, t, s, k) in zip(xs, pix):
        f.node(nid, x, 50, 144, 100, t, s, k)
    for x, (nid, t, s, k) in zip(xs, feat):
        f.node(nid, x, 236, 144, 100, t, s, k)
    for row in (["v1", "v2", "v3", "v4", "v5", "v6", "v7"], ["f1", "f2", "f3", "f4", "f5", "f6", "f7"]):
        for p, q in zip(row, row[1:]):
            f.edge(p, q)
    return f


def calibration_flow(n) -> Flow:
    f = Flow(1160, 400, "Calibration: from scores to a decision policy",
             "Held-out good parts and synthetic defects set the score scale and the cost-optimal threshold.")
    f.node("i1", 20, 24, 220, 78, "Held-out good parts", ["calibration split: " + n["calib_counts"], "parts never seen"], "data")
    f.node("i2", 20, 118, 220, 78, "Synthetic defects", ["full strength + 50% and", "25% replays"], "data")
    f.node("i3", 20, 212, 220, 78, "Diffusion-refined", ["12 Stable-Diffusion", "defects per product"], "data")
    f.node("i4", 20, 306, 220, 78, "Real defects (oracle)", ["35% of the test set —", "comparison only"], "ghost")
    f.node("m1", 280, 118, 200, 78, "Detector raw scores", ["every calibration image", "through the detector"], "model")
    f.node("m2", 510, 118, 190, 78, "0–100 score scale", ["25 = median good part,", "75 = median synthetic"], "proc")
    f.node("m3", 730, 118, 190, 78, "Capability gate", ["a defect set counts only", "if its AUROC ≥ 0.9"], "gate")
    f.node("m4", 950, 104, 196, 106, "Cost search", ["threshold 0–100 (step 0.5)", "× band 0–15; smoothed", "rates; CHECK ≤ 5%"], "proc")
    f.node("o1", 950, 280, 196, 80, "calibration.json", ["default policy (t, band)", "per product + detector"], "store")
    f.node("o2", 730, 306, 190, 78, "Oracle threshold", ["what labels would give:", "1.5–3.5 points away"], "ghost")
    for src in ("i1", "i2", "i3"):
        f.edge(src, "m1")
    f.edge("m1", "m2")
    f.edge("m2", "m3")
    f.edge("m3", "m4")
    f.edge("m4", "o1", sa="b", sb="t", label="deploy")
    f.edge("i4", "o2", dash=True, label="scored the same way")
    f.edge("o2", "o1", dash=True, sa="r", sb=("l", 0.62))
    return f


def monitoring_flow(n) -> Flow:
    f = Flow(1160, 482, "Monitoring and lot change",
             "Camera-health gate, control charts and the lot re-qualification decision.")
    f.node("a", 20, 24, 210, 84, "Every inspected image", ["score + brightness,", "contrast, sharpness"], "data")
    f.node("b", 270, 30, 180, 72, "Health gate", ["any |z| > 4 ?"], "gate")
    f.node("c", 490, 24, 220, 84, "Force CHECK", ["‘check lamp / lens /", "camera’ — no auto PASS"], "alarm")
    f.node("d", 20, 172, 210, 96, "EWMA control charts", ["λ = 0.2, ±3σ limits,", "alarm after 2 in a row;", "PSI on the score histogram"], "proc")
    f.node("e", 270, 184, 180, 72, "Alarm?", ["score or camera signal"], "gate")
    f.node("g", 490, 184, 220, 72, "What moved?", ["camera signals or scores"], "gate")
    f.node("h", 770, 172, 200, 96, "Fix the camera", ["lamp, focus or position;", "re-verify qualified", "parts — no retraining"], "alarm")
    f.node("i", 490, 330, 220, 84, "Sample good parts", ["~10 good parts of the", "current lot (no defects)"], "human")
    f.node("j", 750, 336, 200, 72, "Lot shift test", ["Mann–Whitney U, p < 0.01?"], "gate")
    f.node("k", 990, 324, 160, 96, "Re-qualify", ["+10% coreset of new", "patches; re-fit scale", "and threshold"], "model")
    f.node("l", 760, 434, 180, 40, "Keep the model", [], "proc")
    f.edge("a", "b")
    f.edge("b", "c", label="yes")
    f.edge("a", "d", sa="b", sb="t", label="stream")
    f.edge("d", "e")
    f.edge("e", "g", label="yes")
    f.edge("g", "h", label="camera")
    f.edge("g", "i", sa="b", sb="t", label="only scores of good parts")
    f.edge("i", "j")
    f.edge("j", "k", label="yes")
    f.edge("j", "l", sa="b", sb="t", label="no", lxy=(858, 426, "start"))
    return f


def split_flow(n) -> Flow:
    f = Flow(1160, 360, "The leakage-free split", "Four roles; real defects never train a model or set the deployed threshold.")
    f.node("s1", 20, 40, 240, 92, "MVTec AD · train", ["good parts only:", f"{n['train_total']:,} images (4 products)"], "data")
    f.node("s2", 20, 222, 240, 92, "MVTec AD · test", ["good parts + real defects", f"with masks: {n['test_total']:,} images"], "data")
    rows = [("r1", 12, "① model_train_good", [f"80% · {n['roles']['model_train_good']:,} images"], "model",
             ["builds the memory bank, trains both VAEs", "and computes the mean image"]),
            ("r2", 98, "② calibration_good", [f"20% · {n['roles']['calibration_good']:,} images"], "proc",
             ["with synthetic defects: score scale, thresholds,", "PatchCore recipe and ensemble weight"]),
            ("r3", 192, "③ oracle_calibration", [f"35% of each type · {n['roles']['oracle_calibration']:,}"], "ghost",
             ["the ‘what if we had labels’ threshold (comparison", "only) + current-lot good parts for re-qualification"]),
            ("r4", 278, "④ operating_test", [f"65% of each type · {n['roles']['operating_test']:,}"], "gate",
             ["every reported false-reject, escape and CHECK", "number — never used for any choice"])]
    for nid, y, t, s, k, use in rows:
        f.node(nid, 330, y, 250, 70, t, s, k)
        f.node(nid + "u", 640, y + 8, 500, 54, "", use, "proc")
    for nid, src, lab in (("r1", "s1", "80%"), ("r2", "s1", "20%"), ("r3", "s2", "35%"), ("r4", "s2", "65%")):
        f.edge(src, nid, label=lab)
        f.edge(nid, nid + "u")
    return f


def deployment_flow(n) -> Flow:
    f = Flow(1160, 350, "From repository to users", "How the code and models reach the browser, the API and the plant.")
    f.node("g", 20, 122, 230, 92, "GitHub repository", ["hbasired/defectspotter:", "code, artifacts, reports"], "store")
    f.node("c", 300, 14, 220, 80, "GitHub Actions", ["pytest on every push", "(core + app tests)"], "proc")
    f.node("d", 300, 128, 220, 80, "deploy_hf.py", ["uploads app, package, ui,", "artifacts, reports, docs"], "proc", mono=True)
    f.node("k", 300, 242, 220, 80, "Colab notebook", ["clone → train → charts", "→ launch or deploy"], "proc")
    f.node("h", 580, 128, 230, 80, "Hugging Face Space", ["bhkr0901/defectspotter", "Gradio 6 · ZeroGPU"], "model")
    f.node("u1", 870, 30, 270, 72, "Browser", ["9 tabs, live demo last;", "upload or photograph a part"], "human")
    f.node("u2", 870, 132, 270, 72, "Python · gradio_client", ["/inspect_part, /inspect_batch,", "/health"], "data")
    f.node("u3", 870, 246, 270, 72, "Local or on-premise", ["python app.py — CPU only,", "ONNX Runtime, no PyTorch"], "data")
    f.edge("g", "c", label="push")
    f.edge("g", "d", label="release")
    f.edge("g", "k")
    f.edge("d", "h", label="upload")
    f.edge("k", "h", dash=True, label="optional")
    f.edge("h", "u1", label="web UI")
    f.edge("h", "u2", label="API")
    f.edge("g", "u3", sa="b", sb="l", via=[(None, 338), (846, None), (None, 282)], label="git clone",
           lxy=(560, 332, "middle"))
    return f


def human_flow(n) -> Flow:
    f = Flow(1160, 250, "Human in the loop", "Every CHECK is decided by an inspector and recorded in an audit log.")
    nodes = [("h1", 20, 190, "CHECK decision", ["score inside the band, or", "camera out of range"], "gate"),
             ("h2", 245, 190, "Review queue", ["pending parts with", "heat map and score"], "human"),
             ("h3", 470, 190, "Inspector", ["looks at the part,", "writes a note"], "human"),
             ("h4", 695, 200, "Verdict", ["confirm → REJECT", "release → PASS"], "gate"),
             ("h5", 930, 210, "Audit log (CSV)", ["time, file, detector, score,", "decision, verdict, note"], "store")]
    for nid, x, w, t, s, k in nodes:
        f.node(nid, x, 40, w, 92, t, s, k)
    for p, q in zip("h1 h2 h3 h4".split(), "h2 h3 h4 h5".split()):
        f.edge(p, q)
    f.note(20, 172, "A quality engineer reviews the log (e.g. weekly): a threshold change or a re-qualification is a recorded,")
    f.note(20, 192, "re-validated release. The model is never retrained automatically from verdicts — a feedback loop could")
    f.note(20, 212, "otherwise learn to accept a recurring defect.")
    return f


ALL = {"overview": system_overview, "build": build_pipeline, "inspect": inspection_flow, "patchcore": patchcore_flow,
       "vae": vae_flow, "calibration": calibration_flow, "monitoring": monitoring_flow, "split": split_flow,
       "deployment": deployment_flow, "human": human_flow}
