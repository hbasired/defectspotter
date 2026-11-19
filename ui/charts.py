"""Plotly figure builders for the app (light theme, one y-axis per chart, legends + tables)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from defectspotter.constants import METHOD_LABELS
from defectspotter.viz import HEAT_COLORSCALE

from .theme import (AXIS, CATEGORY_COLORS, CLASS_COLORS, FONT, GRID, INK, INK_2, METHOD_COLORS, MUTED, SERIES,
                    STATUS, STATUS_BG, SURFACE)

BLUE_RAMP = [[0.0, "#f4f8fe"], [0.25, "#b7d3f6"], [0.5, "#6da7ec"], [0.75, "#2a78d6"], [1.0, "#104281"]]


def style(fig: go.Figure, title: str | None = None, height: int = 380, legend: bool = True) -> go.Figure:
    fig.update_layout(
        template="none", height=height, paper_bgcolor="#ffffff", plot_bgcolor=SURFACE,
        font={"family": ", ".join(FONT), "color": INK, "size": 13},
        title={"text": title, "x": 0.01, "xanchor": "left", "font": {"size": 15, "color": INK}} if title else None,
        margin={"l": 56, "r": 18, "t": 56 if title else 24, "b": 92 if legend else 40},
        showlegend=legend,
        legend={"orientation": "h", "yref": "container", "y": 0.0, "yanchor": "bottom", "xanchor": "left", "x": 0.0,
                "font": {"color": INK_2, "size": 12}, "bgcolor": "rgba(0,0,0,0)"},
        hoverlabel={"bgcolor": "#ffffff", "bordercolor": AXIS, "font": {"color": INK}},
        bargap=0.25, bargroupgap=0.08,
    )
    fig.update_xaxes(showgrid=False, linecolor=AXIS, tickcolor=AXIS, tickfont={"color": INK_2}, zeroline=False,
                     title_font={"color": INK_2}, automargin=True)
    fig.update_yaxes(gridcolor=GRID, linecolor=AXIS, tickfont={"color": INK_2}, zeroline=False,
                     title_font={"color": INK_2}, automargin=True)
    return fig


def empty(message: str, height: int = 300) -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=message, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False,
                       font={"color": MUTED, "size": 14})
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return style(fig, height=height, legend=False)


def _zones(fig: go.Figure, threshold: float, band: float, y_label: bool = True) -> None:
    lo, hi = max(0, threshold - band), min(100, threshold + band)
    for x0, x1, name in ((0, lo, "PASS"), (lo, hi, "CHECK"), (hi, 100, "REJECT")):
        if x1 > x0:
            fig.add_vrect(x0=x0, x1=x1, fillcolor=STATUS_BG[name], opacity=0.85, layer="below", line_width=0)
            if y_label:
                fig.add_annotation(x=(x0 + x1) / 2, y=1.0, yref="paper", yanchor="bottom", showarrow=False,
                                   text=f"{ {'PASS': '✓', 'CHECK': '⚠', 'REJECT': '✕'}[name]} {name}",
                                   font={"size": 11, "color": INK_2})
    fig.add_vline(x=threshold, line_color=INK, line_width=1.5)


# ------------------------------------------------------------------ inspection
def decision_bar(score: float, threshold: float, band: float) -> go.Figure:
    """Horizontal 0-100 scale with PASS/CHECK/REJECT zones and the part's score."""
    fig = go.Figure()
    _zones(fig, threshold, band)
    clipped = float(np.clip(score, 0, 100))
    fig.add_trace(go.Scatter(x=[clipped], y=[0.5], mode="markers+text", text=[f"{score:.1f}"],
                             textposition="top center", textfont={"size": 15, "color": INK},
                             marker={"size": 18, "color": INK, "symbol": "diamond", "line": {"width": 2, "color": "#fff"}},
                             hovertemplate="score %{x:.1f}<extra></extra>", showlegend=False))
    fig.update_xaxes(range=[0, 100], title="normalised anomaly score  (25 = typical good part, 75 = typical synthetic defect)",
                     dtick=10)
    fig.update_yaxes(visible=False, range=[0, 1.25])
    return style(fig, height=190, legend=False)


TRANSPARENT_HEAT = [[0.0, "rgba(255,247,188,0)"], [0.3, "rgba(255,247,188,0)"], [0.45, "rgba(254,196,79,0.35)"],
                    [0.6, "rgba(236,112,20,0.6)"], [0.8, "rgba(204,40,40,0.78)"], [1.0, "rgba(120,10,30,0.9)"]]


def anomaly_heatmap(anomaly_map: np.ndarray | None, level: float = 50.0, image: np.ndarray | None = None) -> go.Figure:
    if anomaly_map is None:
        return empty("Inspect a part to see its pixel anomaly map", 420)
    import cv2

    z = np.clip(np.asarray(anomaly_map, np.float32), 0, 100)
    size = 112
    z = cv2.resize(z, (size, size), interpolation=cv2.INTER_AREA)
    fig = go.Figure()
    if image is not None:
        fig.add_trace(go.Image(z=cv2.resize(np.asarray(image), (size, size), interpolation=cv2.INTER_AREA),
                               hoverinfo="skip"))
    fig.add_trace(go.Heatmap(z=z, zmin=0, zmax=100, colorscale=TRANSPARENT_HEAT if image is not None else HEAT_COLORSCALE,
                             colorbar={"title": {"text": "pixel<br>score"}, "thickness": 12},
                             hovertemplate="x %{x}, y %{y}<br>pixel score %{z:.0f}<extra></extra>"))
    fig.add_trace(go.Contour(z=z, contours={"start": level, "end": level, "size": 1, "coloring": "none"},
                             line={"color": "#2a78d6", "width": 2}, showscale=False, hoverinfo="skip",
                             name="unusual-pixel boundary (50)"))
    fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x")
    fig.update_xaxes(visible=False)
    return style(fig, "Pixel anomaly map (hover for values) · blue line = more unusual than 99.9% of good-part pixels",
                 430, legend=False)


# ---------------------------------------------------------------- threshold lab
def score_distribution(groups: dict[str, np.ndarray], threshold: float, band: float, title: str) -> go.Figure:
    fig = go.Figure()
    _zones(fig, threshold, band)
    bins = {"start": 0, "end": 110, "size": 2.5}
    for name, values in groups.items():
        values = np.clip(np.asarray(values, np.float64), 0, 110)
        if len(values) == 0:
            continue
        key = "good" if name.startswith("good") else "synthetic" if "synthetic" in name else (
            "diffusion" if "diffusion" in name else "defect")
        fig.add_trace(go.Histogram(x=values, xbins=bins, name=f"{name} (n={len(values)})", opacity=0.75,
                                   marker={"color": CLASS_COLORS[key], "line": {"width": 1, "color": "#ffffff"}},
                                   histnorm="percent", hovertemplate="%{x}: %{y:.1f}%<extra>" + name + "</extra>"))
    fig.update_layout(barmode="overlay")
    fig.update_xaxes(range=[0, 110], title="normalised anomaly score (values above 110 shown at 110)")
    fig.update_yaxes(title="% of parts in group")
    return style(fig, title, 360)


def rate_curves(curve: pd.DataFrame, threshold: float) -> go.Figure:
    fig = go.Figure()
    for column, label, color in (("false_reject_rate", "False-reject rate (good → REJECT)", SERIES[0]),
                                 ("escape_rate", "Escape rate (defect → PASS)", SERIES[1]),
                                 ("review_rate", "Review load (share of parts → CHECK)", SERIES[2])):
        fig.add_trace(go.Scatter(x=curve.threshold, y=100 * curve[column], mode="lines", name=label,
                                 line={"color": color, "width": 2},
                                 hovertemplate="threshold %{x:.1f}: %{y:.1f}%<extra></extra>"))
    fig.add_vline(x=threshold, line_color=INK, line_width=1.5)
    fig.update_xaxes(title="decision threshold", range=[0, 100])
    fig.update_yaxes(title="rate (%)", rangemode="tozero")
    return style(fig, "Trade-off: moving the threshold trades false rejects for escapes", 340)


def cost_curve(curve: pd.DataFrame, threshold: float, optimum: float, references: dict) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=curve.threshold, y=curve.expected_cost_per_1000, mode="lines",
                             name="DefectSpotter (this threshold)", line={"color": SERIES[0], "width": 2},
                             hovertemplate="threshold %{x:.1f}: %{y:,.0f} per 1,000 parts<extra></extra>"))
    fig.add_hline(y=references["manual_100pct_per_1000"], line_color=MUTED, line_width=1,
                  annotation_text="100% manual inspection", annotation_font_color=INK_2)
    fig.add_hline(y=references["no_inspection_per_1000"], line_color=AXIS, line_width=1,
                  annotation_text="no inspection", annotation_font_color=INK_2)
    fig.add_vline(x=threshold, line_color=INK, line_width=1.5)
    fig.add_trace(go.Scatter(x=[optimum], y=[float(np.interp(optimum, curve.threshold, curve.expected_cost_per_1000))],
                             mode="markers", name="cost-optimal threshold",
                             marker={"size": 11, "color": STATUS["PASS"], "symbol": "star", "line": {"color": "#fff", "width": 1}}))
    fig.update_xaxes(title="decision threshold", range=[0, 100])
    fig.update_yaxes(title="expected cost per 1,000 parts", rangemode="tozero")
    return style(fig, "Expected cost per 1,000 parts on the held-out test line", 340)


# ---------------------------------------------------------------- evaluation
def metric_bars(metrics: pd.DataFrame, metric: str, label: str) -> go.Figure:
    if metrics.empty or metric not in metrics:
        return empty("No evaluation results yet")
    fig = go.Figure()
    for method in ("patchcore", "vae", "mean", "ensemble"):
        part = metrics[metrics.method == method]
        if part.empty:
            continue
        error = None
        if metric == "image_auroc" and {"image_auroc_ci_low", "image_auroc_ci_high"} <= set(part.columns):
            error = {"type": "data", "symmetric": False, "array": part.image_auroc_ci_high - part.image_auroc,
                     "arrayminus": part.image_auroc - part.image_auroc_ci_low, "color": INK_2, "thickness": 1, "width": 3}
        fig.add_trace(go.Bar(x=part.category, y=part[metric], name=METHOD_LABELS[method], error_y=error,
                             marker={"color": METHOD_COLORS[method], "cornerradius": 4},
                             hovertemplate="%{x}: %{y:.3f}<extra>" + METHOD_LABELS[method] + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title=label, range=[0, 1.05] if metric != "image_ap" else [0, 1.05])
    return style(fig, f"{label} by product and detector" + (" (bars: 95% bootstrap CI)" if metric == "image_auroc" else ""), 380)


def roc_chart(roc: dict, category: str) -> go.Figure:
    curves = roc.get(category, {})
    if not curves:
        return empty("No ROC data")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", line={"color": AXIS, "width": 1},
                             name="random guess", hoverinfo="skip"))
    for method in ("patchcore", "vae", "mean", "ensemble"):
        if method in curves and curves[method]["fpr"]:
            fig.add_trace(go.Scatter(x=curves[method]["fpr"], y=curves[method]["tpr"], mode="lines",
                                     name=METHOD_LABELS[method], line={"color": METHOD_COLORS[method], "width": 2},
                                     hovertemplate="FPR %{x:.2f} · TPR %{y:.2f}<extra></extra>"))
    fig.update_xaxes(title="false-positive rate (good parts flagged)", range=[0, 1])
    fig.update_yaxes(title="true-positive rate (defects caught)", range=[0, 1.02])
    return style(fig, f"ROC curves · {category}", 380)


def defect_type_bars(per_defect: pd.DataFrame, category: str, method: str) -> go.Figure:
    part = per_defect[(per_defect.category == category) & (per_defect.method == method)]
    if part.empty:
        return empty("No per-defect data")
    part = part.copy()
    part["label"] = part.defect_type.str.replace("_", " ") + "  (n=" + part.images.astype(str) + ")"
    fig = go.Figure()
    for column, name in (("reject_rate", "REJECT"), ("check_rate", "CHECK"), ("pass_rate", "PASS")):
        fig.add_trace(go.Bar(y=part.label, x=100 * part[column], orientation="h",
                             name=f"{ {'PASS': '✓', 'CHECK': '⚠', 'REJECT': '✕'}[name]} {name}",
                             marker={"color": STATUS[name], "line": {"color": "#ffffff", "width": 2}},
                             hovertemplate="%{y}: %{x:.0f}%<extra>" + name + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="% of test parts (operating split) at the calibrated policy", range=[0, 100])
    fig.update_yaxes(autorange="reversed")
    return style(fig, f"Decisions per defect type · {category} · {METHOD_LABELS[method]}", 120 + 34 * len(part))


def method_scatter(scores: dict, category: str) -> go.Figure:
    data = scores.get(category)
    if not data:
        return empty("No scores")
    labels = np.asarray(data["test_labels"])
    p = np.asarray(data["methods"]["patchcore"]["test"])
    v = np.asarray(data["methods"]["vae"]["test"])
    types = np.asarray(data["test_defect_types"])
    fig = go.Figure()
    for value, name, color in ((0, "good part", CLASS_COLORS["good"]), (1, "defective part", CLASS_COLORS["defect"])):
        m = labels == value
        fig.add_trace(go.Scatter(x=p[m], y=v[m], mode="markers", name=name, text=types[m],
                                 marker={"size": 9, "color": color, "line": {"color": "#ffffff", "width": 1.5}, "opacity": 0.85},
                                 hovertemplate="%{text}<br>PatchCore %{x:.0f} · VAE %{y:.0f}<extra></extra>"))
    fig.update_xaxes(title="PatchCore score")
    fig.update_yaxes(title="VAE score")
    return style(fig, f"Feature memory vs reconstruction · {category} test parts", 380)


def latency_bars(latency: pd.DataFrame) -> go.Figure:
    if latency.empty:
        return empty("No latency measurements")
    fig = go.Figure()
    for method in ("patchcore", "vae", "mean", "ensemble"):
        part = latency[latency.method == method]
        if part.empty:
            continue
        fig.add_trace(go.Bar(x=part.category, y=part.latency_ms_median, name=METHOD_LABELS[method],
                             customdata=part.latency_ms_p95,
                             marker={"color": METHOD_COLORS[method], "cornerradius": 4},
                             hovertemplate="%{x}: median %{y:.0f} ms · p95 %{customdata:.0f} ms<extra></extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="ms per part (median, CPU)")
    return style(fig, "Latency per part, end-to-end (decode → decision)", 340)


def backbone_chart(study: pd.DataFrame) -> go.Figure:
    if study.empty:
        return empty("Backbone study not available")
    agg = study.groupby(["backbone", "image_size"], as_index=False).agg(
        auroc=("image_auroc_topk", "mean"), pixel=("pixel_auroc", "mean"), size=("onnx_mb_fp32_est", "first"),
        ms=("extract_ms_per_image", "mean"))
    agg["name"] = agg.backbone.str.replace("_", " ") + " @" + agg.image_size.astype(str)
    chosen = agg.backbone.eq("resnet18") & agg.image_size.eq(224)
    fig = go.Figure()
    fig.add_vrect(x0=10, x1=20, fillcolor="#eaf7ea", opacity=0.7, line_width=0, layer="below")
    fig.add_annotation(x=np.log10(14), y=1.0, yref="paper", text="target ONNX budget 10–20 MB", showarrow=False,
                       font={"color": INK_2, "size": 11}, yanchor="bottom")
    fig.add_trace(go.Scatter(x=agg["size"], y=agg.auroc, mode="markers+text", text=agg.name,
                             textposition="top center", textfont={"size": 11, "color": INK_2}, name="backbone",
                             customdata=np.c_[agg.pixel, agg.ms],
                             marker={"size": np.where(chosen, 16, 11), "color": np.where(chosen, SERIES[1], SERIES[0]),
                                     "line": {"color": "#ffffff", "width": 2}},
                             hovertemplate="%{text}<br>ONNX %{x:.1f} MB<br>image AUROC %{y:.3f}<br>"
                                           "pixel AUROC %{customdata[0]:.3f}<br>%{customdata[1]:.0f} ms/img<extra></extra>"))
    fig.update_xaxes(type="log", title="feature-extractor ONNX size (MB, log scale)")
    fig.update_yaxes(title="mean image AUROC (bottle, screw, carpet)")
    return style(fig, "Backbone study: accuracy vs deployment size (orange = chosen)", 380, legend=False)


def threshold_dumbbell(thresholds: pd.DataFrame, method: str) -> go.Figure:
    part = thresholds[thresholds.method == method]
    if part.empty:
        return empty("No threshold comparison")
    fig = go.Figure()
    for _, row in part.iterrows():
        fig.add_trace(go.Scatter(x=[row.synthetic_threshold, row.real_threshold], y=[row.category] * 2, mode="lines",
                                 line={"color": AXIS, "width": 3}, showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=part.synthetic_threshold, y=part.category, mode="markers", name="synthetic defects only",
                             marker={"size": 13, "color": SERIES[0], "line": {"color": "#fff", "width": 2}},
                             hovertemplate="%{y}: %{x:.1f}<extra>synthetic</extra>"))
    fig.add_trace(go.Scatter(x=part.real_threshold, y=part.category, mode="markers", name="real labelled defects (oracle)",
                             marker={"size": 13, "color": SERIES[1], "symbol": "diamond", "line": {"color": "#fff", "width": 2}},
                             hovertemplate="%{y}: %{x:.1f}<extra>real</extra>"))
    if "diffusion_threshold" in part and part.diffusion_threshold.notna().any():
        fig.add_trace(go.Scatter(x=part.diffusion_threshold, y=part.category, mode="markers",
                                 name="diffusion-inpainted defects",
                                 marker={"size": 12, "color": SERIES[2], "symbol": "square", "line": {"color": "#fff", "width": 2}},
                                 hovertemplate="%{y}: %{x:.1f}<extra>diffusion</extra>"))
    fig.update_xaxes(title="cost-optimal threshold (0–100 scale)", range=[0, 100])
    return style(fig, f"Threshold chosen without vs with real defects · {METHOD_LABELS[method]}", 300)


# --------------------------------------------------------------------- drift
def drift_stream(stream: pd.DataFrame, category: str, threshold: float, band: float) -> go.Figure:
    part = stream[stream.category == category]
    if part.empty:
        return empty("No drift stream")
    fig = go.Figure()
    for phase, color in (("lamp ageing", "#f4f8fe"), ("after maintenance", "#f6fbf6"), ("camera bumped", "#fff4ec")):
        seg = part[part.phase == phase]
        if not seg.empty:
            fig.add_vrect(x0=seg.t.min() - 0.5, x1=seg.t.max() + 0.5, fillcolor=color, opacity=1, layer="below",
                          line_width=0, annotation_text=phase, annotation_position="top left",
                          annotation_font={"color": INK_2, "size": 11})
    for decision in ("PASS", "CHECK", "REJECT"):
        seg = part[part.decision == decision]
        fig.add_trace(go.Scatter(x=seg.t, y=np.clip(seg.score, 0, 150), mode="markers",
                                 name=f"{ {'PASS': '✓', 'CHECK': '⚠', 'REJECT': '✕'}[decision]} {decision}",
                                 marker={"size": 7, "color": STATUS[decision], "line": {"color": "#fff", "width": 1}},
                                 customdata=seg.is_defect,
                                 hovertemplate="part %{x}: score %{y:.0f}<br>truly defective: %{customdata}<extra></extra>"))
    fig.add_trace(go.Scatter(x=part.t, y=part.score_ewma, mode="lines", name="EWMA of score (winsorised)",
                             line={"color": INK, "width": 2}))
    fig.add_trace(go.Scatter(x=part.t, y=part.score_ucl, mode="lines", name="EWMA control limit",
                             line={"color": SERIES[7], "width": 1.5}))
    alarms = part[part.score_alarm.astype(bool)]
    if not alarms.empty:
        fig.add_trace(go.Scatter(x=alarms.t, y=alarms.score_ewma, mode="markers", name="🔔 drift alarm",
                                 marker={"size": 9, "color": STATUS["REJECT"], "symbol": "x"}))
    fig.update_xaxes(title="part number on the line")
    fig.update_yaxes(title="anomaly score (clipped at 150)")
    return style(fig, "Score stream + EWMA control chart: qualified → lamp ageing → maintenance → camera bumped", 420)


def health_stream(stream: pd.DataFrame, category: str, signal: str) -> go.Figure:
    part = stream[stream.category == category]
    if part.empty or f"{signal}_ewma" not in part:
        return empty("No health signal")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=part.t, y=part[signal], mode="markers", name=signal,
                             marker={"size": 5, "color": SERIES[0], "opacity": 0.5}))
    fig.add_trace(go.Scatter(x=part.t, y=part[f"{signal}_ewma"], mode="lines", name="EWMA",
                             line={"color": INK, "width": 2}))
    for bound in ("ucl", "lcl"):
        fig.add_trace(go.Scatter(x=part.t, y=part[f"{signal}_{bound}"], mode="lines", showlegend=bound == "ucl",
                                 name="control limits", line={"color": SERIES[7], "width": 1.5}))
    fig.update_xaxes(title="part number")
    fig.update_yaxes(title=signal)
    return style(fig, f"Camera health · {signal}", 300)


def drift_heatmap(sensitivity: pd.DataFrame, category: str, column: str, label: str) -> go.Figure:
    part = sensitivity[sensitivity.category == category]
    if part.empty:
        return empty("No drift sensitivity data")
    pivot = part.pivot(index="drift", columns="severity", values=column)
    fig = go.Figure(go.Heatmap(z=100 * pivot.values, x=[f"{s:.0%}" for s in pivot.columns],
                               y=[d.replace("_", " ") for d in pivot.index], colorscale=BLUE_RAMP, zmin=0, zmax=100,
                               text=np.round(100 * pivot.values).astype(int), texttemplate="%{text}%",
                               textfont={"size": 11}, colorbar={"title": {"text": "%"}, "thickness": 12},
                               hovertemplate="%{y} at %{x} severity: %{z:.0f}%<extra></extra>"))
    fig.update_xaxes(title="drift severity", side="bottom")
    return style(fig, label, 330, legend=False)


def sweep_chart(severities, scores, threshold: float, band: float, kind: str) -> go.Figure:
    fig = go.Figure()
    fig.add_hrect(y0=threshold + band, y1=max(110, max(scores) + 5), fillcolor=STATUS_BG["REJECT"], line_width=0, layer="below")
    fig.add_hrect(y0=threshold - band, y1=threshold + band, fillcolor=STATUS_BG["CHECK"], line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=[100 * s for s in severities], y=scores, mode="lines+markers", name="score",
                             line={"color": SERIES[0], "width": 2}, marker={"size": 8}))
    fig.update_xaxes(title=f"{kind.replace('_', ' ')} severity (%)")
    fig.update_yaxes(title="anomaly score")
    return style(fig, "Same part, worsening acquisition: score vs severity", 300, legend=False)


# ------------------------------------------------------------------ few-shot
def few_shot_chart(few: pd.DataFrame) -> go.Figure:
    if few.empty:
        return empty("No few-shot results")
    agg = few.groupby("shots").agg(image=("image_auroc", "mean"), image_lo=("image_auroc", "min"),
                                   image_hi=("image_auroc", "max"), pixel=("pixel_auroc", "mean")).reset_index()
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=list(agg.shots) + list(agg.shots[::-1]), y=list(agg.image_hi) + list(agg.image_lo[::-1]),
                             fill="toself", fillcolor="rgba(42,120,214,0.12)", line={"width": 0}, hoverinfo="skip",
                             name="range over random draws"))
    fig.add_trace(go.Scatter(x=agg.shots, y=agg.image, mode="lines+markers", name="image AUROC",
                             line={"color": SERIES[0], "width": 2}, marker={"size": 8}))
    fig.add_trace(go.Scatter(x=agg.shots, y=agg.pixel, mode="lines+markers", name="pixel AUROC",
                             line={"color": SERIES[1], "width": 2}, marker={"size": 8}))
    fig.update_xaxes(type="log", title="number of good images of the new product (log scale)", tickvals=list(agg.shots))
    fig.update_yaxes(title="AUROC on the new product's test set", range=[0.4, 1.02])
    return style(fig, "Few-shot onboarding: how many good parts does a new product need?", 360)


# ------------------------------------------------------------------- EDA
def composition_chart(composition: pd.DataFrame) -> go.Figure:
    if composition.empty:
        return empty("No dataset composition")
    frame = composition.copy()
    frame["group"] = np.where(frame.split.eq("train"), "train · good",
                              np.where(frame.defect_type.eq("good"), "test · good", "test · defective"))
    agg = frame.groupby(["category", "group"], as_index=False).images.sum()
    fig = go.Figure()
    for i, group in enumerate(("train · good", "test · good", "test · defective")):
        part = agg[agg.group == group]
        fig.add_trace(go.Bar(x=part.category, y=part.images, name=group, text=part.images, textposition="outside",
                             marker={"color": SERIES[i], "cornerradius": 4},
                             hovertemplate="%{x}: %{y} images<extra>" + group + "</extra>"))
    fig.update_layout(barmode="group")
    fig.update_yaxes(title="images")
    return style(fig, "Images per product: training uses good parts only", 360)


def defect_types_chart(composition: pd.DataFrame, category: str) -> go.Figure:
    part = composition[(composition.category == category) & (composition.defect_type != "good")]
    if part.empty:
        return empty("No defect types")
    part = part.groupby("defect_type", as_index=False).images.sum().sort_values("images")
    fig = go.Figure(go.Bar(x=part.images, y=part.defect_type.str.replace("_", " "), orientation="h",
                           marker={"color": CATEGORY_COLORS.get(category, SERIES[0]), "cornerradius": 4},
                           text=part.images, textposition="outside",
                           hovertemplate="%{y}: %{x} test images<extra></extra>"))
    fig.update_xaxes(title="labelled test images")
    return style(fig, f"Defect types · {category}", 120 + 36 * len(part), legend=False)


def acquisition_box(stats: pd.DataFrame, signal: str) -> go.Figure:
    if stats.empty or signal not in stats:
        return empty("No image statistics")
    fig = go.Figure()
    for category in stats.category.unique():
        part = stats[stats.category == category]
        fig.add_trace(go.Box(y=part[signal], name=category, marker={"color": CATEGORY_COLORS.get(category, SERIES[0]), "size": 4},
                             line={"width": 1.5}, boxpoints="outliers"))
    fig.update_yaxes(title=signal.replace("_", " "))
    return style(fig, f"Acquisition signal by product · {signal.replace('_', ' ')}", 340, legend=False)


def mask_area_chart(stats: pd.DataFrame) -> go.Figure:
    part = stats[(stats.label == 1) & (stats.mask_area_pct > 0)]
    if part.empty:
        return empty("No defect masks")
    fig = go.Figure()
    for category in part.category.unique():
        values = part[part.category == category].mask_area_pct
        fig.add_trace(go.Box(x=values, name=category, orientation="h", boxpoints="all", jitter=0.4, pointpos=0,
                             marker={"color": CATEGORY_COLORS.get(category, SERIES[0]), "size": 5, "opacity": 0.6},
                             line={"width": 1.5}, hovertemplate="%{x:.2f}% of image<extra>" + category + "</extra>"))
    fig.update_xaxes(type="log", title="defect area (% of image, log scale)")
    return style(fig, "Defect size: many defects cover < 1% of the image", 320, legend=False)


def batch_strip(frame: pd.DataFrame, threshold: float, band: float) -> go.Figure:
    if frame.empty:
        return empty("Run a batch to see the line view")
    data = frame.reset_index(drop=True)
    fig = go.Figure()
    fig.add_hrect(y0=threshold + band, y1=max(110, data.score.max() + 5), fillcolor=STATUS_BG["REJECT"], line_width=0, layer="below")
    fig.add_hrect(y0=threshold - band, y1=threshold + band, fillcolor=STATUS_BG["CHECK"], line_width=0, layer="below")
    for decision in ("PASS", "CHECK", "REJECT"):
        part = data[data.decision == decision]
        fig.add_trace(go.Scatter(x=part.index + 1, y=part.score, mode="markers",
                                 name=f"{ {'PASS': '✓', 'CHECK': '⚠', 'REJECT': '✕'}[decision]} {decision}",
                                 text=part.file, marker={"size": 11, "color": STATUS[decision], "line": {"color": "#fff", "width": 1.5}},
                                 hovertemplate="%{text}<br>score %{y:.1f}<extra></extra>"))
    fig.update_xaxes(title="part in batch")
    fig.update_yaxes(title="anomaly score")
    return style(fig, "Batch scores against the decision zones", 330)


def vae_history_chart(history: pd.DataFrame) -> go.Figure:
    if history.empty:
        return empty("No VAE training history")
    fig = go.Figure()
    for category in history.category.unique():
        part = history[history.category == category]
        fig.add_trace(go.Scatter(x=part.epoch, y=part.val_loss, mode="lines", name=f"{category} validation",
                                 line={"color": CATEGORY_COLORS.get(category, SERIES[0]), "width": 2}))
    fig.update_xaxes(title="epoch")
    fig.update_yaxes(title="validation loss (L1 + SSIM)")
    return style(fig, "VAE training on good parts only", 320)


def cost_comparison(costs: dict, ours: float) -> go.Figure:
    labels = ["No inspection", "100% manual inspection", "DefectSpotter + human CHECK"]
    values = [costs["no_inspection_per_1000"], costs["manual_100pct_per_1000"], ours]
    fig = go.Figure(go.Bar(x=values, y=labels, orientation="h", text=[f"{v:,.0f}" for v in values],
                           textposition="outside", marker={"color": [AXIS, MUTED, SERIES[0]], "cornerradius": 4},
                           hovertemplate="%{y}: %{x:,.0f} per 1,000 parts<extra></extra>"))
    fig.update_xaxes(title="expected cost per 1,000 parts (illustrative units)")
    fig.update_yaxes(autorange="reversed")
    return style(fig, "Business case at the default cost model", 260, legend=False)


def accuracy_latency(metrics: pd.DataFrame, latency: pd.DataFrame) -> go.Figure:
    if metrics.empty or latency.empty:
        return empty("No latency/accuracy data")
    frame = metrics.merge(latency, on=["category", "method"])
    fig = go.Figure()
    for method in ("patchcore", "vae", "mean", "ensemble"):
        part = frame[frame.method == method]
        if part.empty:
            continue
        fig.add_trace(go.Scatter(x=part.latency_ms_median, y=part.image_auroc, mode="markers+text", text=part.category,
                                 textposition="top center", textfont={"size": 11, "color": INK_2},
                                 name=METHOD_LABELS[method],
                                 marker={"size": 12, "color": METHOD_COLORS[method], "line": {"color": "#fff", "width": 2}},
                                 hovertemplate="%{text}: %{y:.3f} AUROC at %{x:.0f} ms<extra>" + METHOD_LABELS[method] + "</extra>"))
    fig.update_xaxes(title="median latency per part (ms, CPU)", rangemode="tozero")
    fig.update_yaxes(title="image AUROC", range=[0, 1.05])
    return style(fig, "Accuracy vs latency (top-left is best)", 380)


def footprint_chart(sizes: pd.DataFrame) -> go.Figure:
    if sizes.empty:
        return empty("No artifact sizes")
    frame = sizes.copy()
    frame["component"] = np.select(
        [frame.file.str.endswith("feature_extractor.onnx"), frame.file.str.contains("patchcore_bank"),
         frame.file.str.endswith("vae.onnx"), frame.file.str.contains("mean_baseline")],
        ["shared CNN extractor (ONNX)", "PatchCore memory bank", "VAE (ONNX)", "mean-image baseline"], "metadata")
    agg = frame.groupby(["group", "component"], as_index=False).size_mb.sum()
    fig = go.Figure()
    for i, component in enumerate(["shared CNN extractor (ONNX)", "PatchCore memory bank", "VAE (ONNX)",
                                   "mean-image baseline"]):
        part = agg[agg.component == component]
        if part.empty:
            continue
        fig.add_trace(go.Bar(y=part.group, x=part.size_mb, orientation="h", name=component,
                             marker={"color": SERIES[i], "line": {"color": "#ffffff", "width": 2}},
                             hovertemplate="%{y}: %{x:.1f} MB<extra>" + component + "</extra>"))
    fig.update_layout(barmode="stack")
    fig.update_xaxes(title="MB on disk (budget: extractor 10-20 MB, bank < 50 MB per product)")
    return style(fig, f"Deployment footprint · total {sizes.size_mb.sum():.0f} MB", 300)


def psi_chart(stream: pd.DataFrame, category: str, reference: np.ndarray, window: int = 30) -> go.Figure:
    from defectspotter.drift import population_stability_index

    part = stream[stream.category == category]
    if part.empty or len(reference) == 0:
        return empty("No PSI data")
    scores = part.score.to_numpy()
    psi = [population_stability_index(reference, scores[max(0, i - window + 1): i + 1]) if i + 1 >= 10 else np.nan
           for i in range(len(scores))]
    fig = go.Figure()
    fig.add_hrect(y0=0.1, y1=0.25, fillcolor=STATUS_BG["CHECK"], line_width=0, layer="below")
    fig.add_hrect(y0=0.25, y1=max(0.3, np.nanmax(psi) * 1.05), fillcolor=STATUS_BG["REJECT"], line_width=0, layer="below")
    fig.add_trace(go.Scatter(x=part.t, y=psi, mode="lines", name=f"PSI (last {window} parts vs qualified good parts)",
                             line={"color": SERIES[0], "width": 2}))
    fig.update_xaxes(title="part number")
    fig.update_yaxes(title="population stability index", rangemode="tozero")
    return style(fig, "Score-distribution shift (PSI): < 0.1 stable · 0.1–0.25 warn · > 0.25 alert", 300)
