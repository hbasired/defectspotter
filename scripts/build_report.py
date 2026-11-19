"""Build a static, single-file HTML presentation of the project from reports/ (no server needed).

    python scripts/build_report.py            # -> reports/DefectSpotter_report.html

The page reuses the app's story sections and Plotly charts; figures are embedded as base64 so the file can be
opened offline, attached to an e-mail or hosted as a static page.
"""
from __future__ import annotations

import base64
import html
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from ui import charts  # noqa: E402
from ui.app import MONITORING_POLICY, _eda_insights  # noqa: E402
from ui.store import ReportStore  # noqa: E402
from ui.story import (cost_summary, data_html, hero_html, pipeline_html, problem_html, responsible_html,  # noqa: E402
                      results_html)
from ui.theme import CSS, INK, INK_2, PAGE  # noqa: E402

import plotly.offline as _plotly_offline  # noqa: E402

# Load the exact plotly.js version that matches the installed plotly Python package.
PLOTLY_CDN = f"https://cdnjs.cloudflare.com/ajax/libs/plotly.js/{_plotly_offline.get_plotlyjs_version()}/plotly.min.js"


def fig_html(fig) -> str:
    return '<div class="ds-chart">' + fig.to_html(include_plotlyjs=False, full_html=False,
                                                   config={"displaylogo": False, "responsive": True}) + "</div>"


def img_html(path: str | None, caption: str, width: str = "100%") -> str:
    if not path or not Path(path).exists():
        return ""
    data = base64.b64encode(Path(path).read_bytes()).decode()
    mime = "image/png" if path.endswith(".png") else "image/jpeg"
    return (f'<figure><img src="data:{mime};base64,{data}" style="width:{width};border-radius:10px" '
            f'alt="{html.escape(caption)}"><figcaption class="ds-small">{html.escape(caption)}</figcaption></figure>')


def grid(*items: str, cols: int = 2) -> str:
    return f'<div class="ds-grid{cols}">' + "".join(f"<div>{i}</div>" for i in items if i) + "</div>"


def markdown_table(md: str) -> str:
    """Tiny Markdown-table/bullets renderer for the monitoring policy and EDA insights."""
    out, rows = [], []
    for line in md.strip().splitlines():
        if line.startswith("|"):
            cells = [c.strip() for c in line.strip("|").split("|")]
            if not all(set(c) <= {"-", ":"} for c in cells):
                rows.append(cells)
            continue
        if rows:
            out.append("<table class='ds-table'>" + "".join(
                "<tr>" + "".join(f"<{'th' if i == 0 else 'td'}>{html.escape(c).replace('**', '')}</{'th' if i == 0 else 'td'}>"
                                 for c in r) + "</tr>" for i, r in enumerate(rows)) + "</table>")
            rows = []
        if line.startswith("### "):
            out.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("- "):
            text = html.escape(line[2:])
            while "**" in text:
                text = text.replace("**", "<b>", 1).replace("**", "</b>", 1)
            out.append(f"<li>{text}</li>")
        elif line.strip():
            out.append(f"<p class='ds-note'>{html.escape(line)}</p>")
    if rows:
        out.append("<table class='ds-table'>" + "".join("<tr>" + "".join(f"<td>{html.escape(c)}</td>" for c in r) + "</tr>"
                                                       for r in rows) + "</table>")
    return "\n".join(out)


def main() -> None:
    store = ReportStore(ROOT)
    metrics = store.csv("metrics.csv")
    categories = list(metrics.category.unique()) if not metrics.empty else []
    comp, stats = store.csv("dataset_composition.csv"), store.csv("image_stats.csv")
    stream, sens = store.csv("drift_stream.csv"), store.csv("drift_sensitivity.csv")
    thresholds = store.csv("threshold_comparison.csv")
    import json

    policies = {c: json.loads((ROOT / "artifacts" / c / "calibration.json").read_text())["methods"]["patchcore"]
                ["policies"]["synthetic_cost"] for c in categories}
    cs = cost_summary(store)
    sections = [
        ("story", "Story", hero_html(store, categories) + problem_html() + pipeline_html()),
        ("data", "Data & EDA", data_html(store) + grid(fig_html(charts.composition_chart(comp)),
                                                        fig_html(charts.mask_area_chart(stats)))
         + grid(*[fig_html(charts.acquisition_box(stats, s)) for s in ("brightness", "contrast", "sharpness")], cols=3)
         + grid(*[fig_html(charts.defect_types_chart(comp, c)) for c in categories], cols=3)
         + grid(*[img_html(store.figure(f"eda_montage_{c}.jpg"), f"{c}: good parts and every defect type") for c in categories], cols=3)
         + f"<div class='ds-card'>{markdown_table(_eda_insights(stats, comp))}</div>"),
        ("results", "Results", results_html(store)
         + grid(fig_html(charts.metric_bars(metrics, "image_auroc", "Image AUROC")),
                fig_html(charts.metric_bars(metrics, "pixel_auroc", "Pixel AUROC")))
         + grid(*[fig_html(charts.roc_chart(store.json("roc_curves.json"), c)) for c in categories], cols=3)
         + grid(*[fig_html(charts.method_scatter(store.json("scores.json"), c)) for c in categories], cols=3)
         + "".join(img_html(store.figure(f"qualitative_{c}.jpg"), f"{c}: input | ground truth | PatchCore | VAE | mean baseline",
                            "80%") for c in categories)),
        ("thresholds", "Thresholds & cost", grid(fig_html(charts.threshold_dumbbell(thresholds, "patchcore")),
                                                 fig_html(charts.threshold_dumbbell(thresholds, "vae")))
         + grid(fig_html(charts.cost_comparison(store.json("cost_model.json"), cs["ours"]))
                if np.isfinite(cs["ours"]) else "",
                *[fig_html(charts.defect_type_bars(store.csv("per_defect_type.csv"), c, "patchcore")) for c in categories[:1]])
         + grid(*[fig_html(charts.defect_type_bars(store.csv("per_defect_type.csv"), c, "patchcore")) for c in categories[1:]])),
        ("engineering", "Deployment", grid(fig_html(charts.backbone_chart(store.csv("backbone_study.csv"))),
                                           fig_html(charts.accuracy_latency(metrics, store.csv("latency.csv"))))
         + grid(fig_html(charts.latency_bars(store.csv("latency.csv"))),
                fig_html(charts.footprint_chart(store.csv("artifact_sizes.csv"))))),
        ("drift", "Drift", "".join(fig_html(charts.drift_stream(stream, c, policies[c]["threshold"], policies[c]["band"]))
                                   for c in categories)
         + grid(*[fig_html(charts.drift_heatmap(sens, c, "good_false_reject_rate", f"False rejects on good parts · {c}"))
                  for c in categories], cols=3)
         + grid(*[fig_html(charts.drift_heatmap(sens, c, "health_flag_rate", f"Caught by camera-health gate · {c}"))
                  for c in categories], cols=3)
         + f"<div class='ds-card'>{markdown_table(MONITORING_POLICY)}</div>"),
        ("fewshot", "New product", fig_html(charts.few_shot_chart(store.csv("few_shot.csv")))),
        ("responsible", "Responsible AI", responsible_html()),
    ]
    nav = "".join(f'<a href="#{key}">{title}</a>' for key, title, _ in sections)
    body = "".join(f'<section id="{key}">{content}</section>' for key, _, content in sections)
    page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DefectSpotter report</title>
<script src="{PLOTLY_CDN}"></script>
<style>
{CSS}
body {{ margin:0; background:{PAGE}; color:{INK}; font-family:system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }}
main {{ max-width:1280px; margin:0 auto; padding:16px; }}
nav {{ position:sticky; top:0; z-index:5; background:rgba(245,246,248,.94); backdrop-filter:blur(6px);
       border-bottom:1px solid rgba(11,11,11,.08); padding:10px 16px; display:flex; gap:16px; flex-wrap:wrap; }}
nav a {{ color:{INK_2}; text-decoration:none; font-weight:600; font-size:.92rem; }}
nav a:hover {{ color:#1c5cab; }}
section {{ margin:18px 0 28px; scroll-margin-top:60px; display:flex; flex-direction:column; gap:14px; }}
.ds-chart {{ background:#fff; border:1px solid rgba(11,11,11,.08); border-radius:14px; padding:6px; overflow:hidden; }}
figure {{ margin:0; }}
</style></head>
<body><nav>{nav}</nav><main>{body}
<p class="ds-small">Generated from the run's reports/ folder. MVTec AD images © MVTec Software GmbH, CC BY-NC-SA 4.0 —
research and teaching only.</p></main></body></html>"""
    out = ROOT / "reports" / "DefectSpotter_report.html"
    out.write_text(page, encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size / 1024 ** 2:.1f} MB)")


if __name__ == "__main__":
    main()
