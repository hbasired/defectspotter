"""Build docs/DefectSpotter_Complete_Guide.html — the complete, self-contained HTML guide.

Every number is read from reports/ and artifacts/ (the same files the app reads), so the guide never disagrees with the
deployed models. Screenshots come from docs/screenshots/ (WebP); figures from reports/figures/. One file, no external
requests: open it in any browser, e-mail it, or print it to PDF.

    python scripts/build_guide.py                 # -> docs/DefectSpotter_Complete_Guide.html
    python scripts/build_guide.py --out guide.html
"""
from __future__ import annotations

import argparse
import datetime as dt
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from guide_html import content_1, content_2, helpers, svg  # noqa: E402
from guide_html.data import POLICY_LABELS, GuideData, raw_uri  # noqa: E402
from guide_html.style import CSS, JS  # noqa: E402


def deployed_rates(d: GuideData) -> dict:
    from defectspotter.operations import rates_from_scores
    p = float(d.cost["defect_prevalence"])
    out = {}
    for cat in d.cats:
        dep = d.deployed(cat)
        if dep["lot"] and cat in d.req_scores:
            s = d.req_scores[cat]
            r = rates_from_scores(s["operating_scores_requalified"], s["operating_labels"], dep["threshold"], dep["band"])
            labels = list(s["operating_labels"])
            r.update(n_good=labels.count(0), n_defect=labels.count(1), model="re-qualified for the current lot")
        else:
            row = d.op(cat)
            r = {k: float(row[k]) for k in ("false_reject_rate", "good_check_rate", "escape_rate", "defect_check_rate",
                                             "defect_reject_rate")}
            r.update(n_good=int(row.n_good), n_defect=int(row.n_defect),
                     model="qualification model" + (" (pose-normalised)" if d.summary["patchcore_configs"][cat].get("align") else ""))
        r["check_rate"] = (1 - p) * r["good_check_rate"] + p * r["defect_check_rate"]
        out[cat] = r
    return out


def api_example(root: Path) -> dict:
    from defectspotter.data import read_rgb
    from defectspotter.runtime import Inspector
    inspector = Inspector(root / "artifacts")
    path = root / "demo_images" / "screw" / "defect_scratch_head_009.png"
    r = inspector.inspect(read_rgb(path), "screw", "patchcore", compare=True, lot=inspector.shipped_lot("screw"))
    regions = [{k: (round(float(v), 2) if isinstance(v, float) else v) for k, v in row.items()}
               for row in r["regions"].to_dict(orient="records")]
    health = [{"signal": h["signal"], "value": round(h["value"], 1), "reference_mean": round(h["reference_mean"], 1),
               "z": round(h["z"], 2), "status": h["status"]} for h in r["health"]]
    return {"decision": r["decision"], "score": round(r["score"], 2), "threshold": r["threshold"], "band": r["band"],
            "category": "screw", "method": "patchcore", "policy": r["policy"], "regions": regions, "health": health,
            "detectors": [{"method": x["method"], "score": x["score"], "decision": x["decision"]} for x in r["comparison"]],
            "latency_ms": round(r["timing"]["total_ms"], 1)}


def context(root: Path) -> SimpleNamespace:
    from defectspotter.data import read_rgb
    from defectspotter.operations import manual_inspection_view, shift_view
    from defectspotter.pose import align_image
    from ui import explain as ex
    from ui.store import ReportStore

    d = GuideData(root)
    rows, scenario = d.real_world()
    views = [(p, m, shift_view(r, scenario)) for p, m, r in rows]
    manual = manual_inspection_view(scenario)
    deployed_views = [v for p, m, v in views if "before" not in m]
    saving = 100 * sum(1 - v["cost_per_shift"] / manual["cost_per_shift"] for v in deployed_views) / max(1, len(deployed_views))
    shots = root / "docs" / "screenshots"
    return SimpleNamespace(
        d=d, n=d.numbers(), root=root, scenario=scenario, manual=manual, rw_rows=rows, rw_views=views,
        saving_vs_manual=saving, deployed_rates=deployed_rates(d),
        detector_stats=ex.detector_stats(ReportStore(root), SimpleNamespace(categories=d.cats)),
        api_example=api_example(root),
        dec=helpers.dec, policy_label=lambda name: POLICY_LABELS.get(name, name),
        shot=lambda name: raw_uri(shots / f"{name}.webp"),
        figure=lambda name, max_w=1400: helpers.img_uri(root / "reports" / "figures" / name, max_w),
        thumb=lambda src, size=220: helpers.thumb_uri(src, size),
        read_rgb=read_rgb, align=align_image)


def toc_html() -> str:
    out = []
    for level, cid, title in helpers.TOC:
        if level == 0:
            out.append(f'<div class="part">{helpers.e(title)}</div>')
        elif level == 1:
            out.append(f'<a href="#{cid}">{helpers.e(title)}</a>')
        else:
            out.append(f'<a class="sub" href="#{cid}">{helpers.e(title)}</a>')
    return "".join(out)


def build(root: Path, out: Path) -> Path:
    helpers.TOC.clear()
    c = context(root)
    today = dt.date.today().isoformat()
    body = []
    for part_title, chapters in content_1.CHAPTERS + content_2.CHAPTERS:
        body.append(helpers.part(part_title))
        body.extend(ch(c) for ch in chapters)
    hero = f"""
<div class="hero" id="top">
 <div class="eyebrow">Complete guide · version 3 · {today}</div>
 <h1>DefectSpotter</h1>
 <p class="lead">Visual inspection that learns only from good parts: what it does, how every part of it works, every model,
 concept and parameter, the results, and how to run and present it.</p>
 <div class="links">
  <a class="btn primary" href="https://bhkr0901-defectspotter.hf.space/">Open the live app ↗</a>
  <a class="btn" href="https://huggingface.co/spaces/bhkr0901/defectspotter">Hugging Face Space ↗</a>
  <a class="btn" href="https://github.com/hbasired/defectspotter">GitHub repository ↗</a>
 </div>
 <div class="meta">Generated by <code>scripts/build_guide.py</code> from <code>reports/</code> and <code>artifacts/</code> — every
 number is the one the deployed models produce. Self-contained: works offline, prints to PDF.</div>
</div>"""
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DefectSpotter Complete Guide</title>
<meta name="description" content="DefectSpotter — the complete guide: flows, data, models, calibration, results, app and demo.">
<style>{CSS}</style><script>{JS}</script></head>
<body>
<header class="topbar"><button id="tocbtn" class="btn" type="button">☰ Contents</button>
<div class="brand">DefectSpotter <span>· complete guide</span></div><div class="spacer"></div>
<a class="btn" href="https://bhkr0901-defectspotter.hf.space/">Live app ↗</a>
<button id="themebtn" class="btn" type="button">☾ Dark</button><div class="progress"></div></header>
<div class="layout">
<nav id="toc" aria-label="Contents"><input id="tocsearch" type="search" placeholder="Search the guide…" aria-label="Search the guide">
<div id="tochint" class="hint"></div>{toc_html()}</nav>
<main><div class="wrap">{hero}{"".join(body)}</div></main></div>
<a class="btn backtop" href="#top">↑ Top</a>
<div id="lightbox" aria-hidden="true"><img alt=""></div>
</body></html>"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default=str(ROOT / "docs" / "DefectSpotter_Complete_Guide.html"))
    args = parser.parse_args()
    path = build(Path(args.root), Path(args.out))
    warnings = [w for f in svg.ALL.values() for w in f(GuideData(Path(args.root)).numbers()).warnings]
    for w in warnings:
        print("  layout warning:", w)
    print(f"wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {sum(1 for lvl, *_ in helpers.TOC if lvl == 1)} chapters)")


if __name__ == "__main__":
    main()
