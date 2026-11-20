"""Light theme, CSS and the force-light script for the DefectSpotter app."""
from __future__ import annotations

import gradio as gr

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
SURFACE = "#fcfcfb"
PAGE = "#f5f6f8"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots (validated: adjacent CVD dE >= 9.1, first three all-pairs safe).
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_COLORS = {"patchcore": SERIES[0], "fvae": SERIES[6], "vae": SERIES[1], "mean": SERIES[2], "ensemble": SERIES[3]}
CATEGORY_COLORS = {"bottle": SERIES[0], "screw": SERIES[1], "carpet": SERIES[2], "metal_nut": SERIES[6]}
CLASS_COLORS = {"good": SERIES[0], "defect": SERIES[1], "synthetic": SERIES[2], "diffusion": SERIES[6]}

# Status palette - reserved for decisions, always shown with an icon + label.
STATUS = {"PASS": "#0ca30c", "CHECK": "#fab219", "REJECT": "#d03b3b"}
STATUS_TEXT = {"PASS": "#006300", "CHECK": "#7a5200", "REJECT": "#9b1c1c"}
STATUS_BG = {"PASS": "#eaf7ea", "CHECK": "#fff6dc", "REJECT": "#fdeceb"}
STATUS_ICON = {"PASS": "✓", "CHECK": "⚠", "REJECT": "✕"}

FONT = ["system-ui", "-apple-system", "Segoe UI", "Roboto", "Helvetica Neue", "Arial", "sans-serif"]


def build_theme() -> gr.themes.Base:
    theme = gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.orange,
                           neutral_hue=gr.themes.colors.slate, font=FONT,
                           font_mono=["ui-monospace", "SFMono-Regular", "Menlo", "monospace"])
    theme = theme.set(
        body_background_fill=PAGE, body_text_color=INK, body_text_color_subdued=INK_2,
        background_fill_primary="#ffffff", background_fill_secondary=SURFACE,
        block_background_fill="#ffffff", block_border_color="rgba(11,11,11,0.09)", block_border_width="1px",
        block_radius="14px", block_shadow="0 1px 2px rgba(15,23,42,0.04)",
        block_label_background_fill="#ffffff", block_label_text_color=INK_2, block_title_text_color=INK,
        input_background_fill="#ffffff", input_border_color="rgba(11,11,11,0.14)",
        button_primary_background_fill="#2a78d6", button_primary_background_fill_hover="#256abf",
        button_primary_text_color="#ffffff", button_secondary_background_fill="#ffffff",
        button_secondary_border_color="rgba(11,11,11,0.14)", button_secondary_text_color=INK,
        slider_color="#2a78d6", color_accent_soft="#e8f1fc", border_color_primary="rgba(11,11,11,0.10)",
        table_even_background_fill="#ffffff", table_odd_background_fill="#fafafa",
        panel_background_fill="#ffffff", code_background_fill="#f3f4f6",
    )
    # Make the dark-mode variables identical to the light ones: the app is light-only.
    for attribute in list(vars(theme)):
        if attribute.endswith("_dark"):
            base = attribute[: -len("_dark")]
            if hasattr(theme, base):
                setattr(theme, attribute, getattr(theme, base))
    return theme


FORCE_LIGHT_JS = """
() => {
  const strip = () => { document.documentElement.classList.remove('dark'); document.body && document.body.classList.remove('dark'); };
  strip();
  new MutationObserver(strip).observe(document.documentElement, {attributes: true, subtree: false, attributeFilter: ['class']});
  if (document.body) new MutationObserver(strip).observe(document.body, {attributes: true, attributeFilter: ['class']});
}
"""

CSS = f"""
:root {{ color-scheme: light; }}
.gradio-container {{ max-width: 1480px !important; margin: 0 auto !important; background: {PAGE}; }}
.ds-hero {{ border-radius: 20px; padding: 26px 30px 22px; margin-bottom: 6px;
  background: linear-gradient(120deg, #eef4ff 0%, #ffffff 48%, #fff4ec 100%);
  border: 1px solid rgba(11,11,11,0.08); }}
.ds-hero h1 {{ margin: 0 0 6px; font-size: 2.1rem; letter-spacing: -0.02em; color: {INK}; }}
.ds-hero .tag {{ font-size: 1.04rem; color: {INK_2}; margin: 0 0 12px; max-width: 980px; }}
.ds-chip {{ display: inline-block; padding: 4px 10px; margin: 0 6px 6px 0; border-radius: 999px;
  border: 1px solid rgba(42,120,214,0.25); background: #ffffff; color: #1c5cab; font-size: .82rem; font-weight: 600; }}
.ds-kpis {{ display: grid; grid-template-columns: repeat(6, minmax(0, 1fr)); gap: 10px; margin-top: 14px; }}
.ds-kpi {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 14px; padding: 12px 14px; }}
.ds-kpi .v {{ font-size: 1.45rem; font-weight: 750; color: {INK}; line-height: 1.15; }}
.ds-kpi .l {{ font-size: .78rem; color: {INK_2}; margin-top: 3px; line-height: 1.25; }}
.ds-kpi .s {{ font-size: .72rem; color: {MUTED}; margin-top: 2px; }}
.ds-card {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 16px; padding: 18px 20px; }}
.ds-card h3 {{ margin-top: 0; }}
.ds-grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }}
.ds-grid3 {{ display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 14px; }}
.ds-grid4 {{ display: grid; grid-template-columns: repeat(4, minmax(0,1fr)); gap: 12px; }}
.ds-step {{ border-left: 4px solid #2a78d6; background: #f4f8fe; border-radius: 0 12px 12px 0; padding: 10px 14px; margin: 8px 0; }}
.ds-step b {{ color: #1c5cab; }}
.ds-note {{ color: {INK_2}; font-size: .9rem; }}
.ds-small {{ color: {MUTED}; font-size: .8rem; }}
.ds-flow {{ display: flex; flex-wrap: wrap; align-items: stretch; gap: 8px; margin: 10px 0 4px; }}
.ds-flow .box {{ flex: 1 1 140px; background: #ffffff; border: 1px solid rgba(11,11,11,0.10); border-radius: 12px;
  padding: 10px 12px; font-size: .86rem; color: {INK_2}; }}
.ds-flow .box b {{ display: block; color: {INK}; font-size: .92rem; margin-bottom: 2px; }}
.ds-flow .arrow {{ align-self: center; color: {MUTED}; font-size: 1.3rem; }}
.ds-decision {{ border-radius: 16px; padding: 16px 18px; border: 1px solid; }}
.ds-decision .big {{ font-size: 1.9rem; font-weight: 800; letter-spacing: .02em; }}
.ds-decision .meta {{ font-size: .92rem; margin-top: 4px; }}
.ds-decision .why {{ font-size: .86rem; margin-top: 8px; }}
.ds-badge {{ display: inline-block; border-radius: 8px; padding: 2px 8px; font-weight: 700; font-size: .8rem; }}
.ds-table {{ width: 100%; border-collapse: collapse; font-size: .88rem; }}
.ds-table th {{ text-align: left; color: {INK_2}; font-weight: 600; border-bottom: 1px solid {GRID}; padding: 6px 8px; }}
.ds-table td {{ border-bottom: 1px solid #f0efec; padding: 6px 8px; font-variant-numeric: tabular-nums; }}
.ds-warn {{ background: #fff6dc; border: 1px solid #f3d27a; color: #7a5200; border-radius: 12px; padding: 10px 14px; }}
.ds-ok {{ background: #eaf7ea; border: 1px solid #b6e2b6; color: #006300; border-radius: 12px; padding: 10px 14px; }}
.ds-section-title {{ font-size: 1.25rem; font-weight: 750; margin: 6px 0 2px; color: {INK}; }}
.tab-nav button {{ font-weight: 600 !important; }}
.ds-kpis.ds-kpis8 {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.ds-hero .ds-kpis {{ grid-template-columns: repeat(4, minmax(0, 1fr)); }}
.ds-intro {{ background: linear-gradient(90deg, #f4f8fe, #ffffff); border: 1px solid rgba(42,120,214,0.18);
  border-left: 5px solid #2a78d6; border-radius: 14px; padding: 14px 18px; margin: 4px 0 10px; }}
.ds-intro-title {{ font-size: 1.22rem; font-weight: 760; color: {INK}; margin-bottom: 4px; }}
.ds-intro-what {{ color: {INK_2}; font-size: .95rem; line-height: 1.45; }}
.ds-intro-how {{ color: {INK_2}; font-size: .88rem; margin-top: 6px; padding-top: 6px; border-top: 1px dashed rgba(11,11,11,0.10); }}
.ds-card-title {{ font-size: 1.08rem; font-weight: 740; color: {INK}; margin-bottom: 8px; }}
.ds-subcard {{ background: {SURFACE}; border: 1px solid rgba(11,11,11,0.07); border-radius: 12px; padding: 12px 14px; font-size: .9rem; color: {INK_2}; }}
.ds-subcard b {{ color: {INK}; }}
.ds-caption {{ color: {INK_2}; font-size: .84rem; line-height: 1.4; background: #fbfbfa; border: 1px solid rgba(11,11,11,0.06);
  border-radius: 10px; padding: 8px 12px; margin: -2px 0 8px; }}
.ds-caption b {{ color: {INK}; }}
.ds-nav {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 16px; padding: 10px 14px; margin: 8px 0 4px; }}
.ds-nav-title {{ font-size: .82rem; color: {MUTED}; font-weight: 650; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 6px; }}
.ds-nav-row {{ display: grid; grid-template-columns: repeat(9, minmax(0, 1fr)); gap: 6px; }}
.ds-nav-step {{ display: flex; gap: 6px; align-items: flex-start; background: {SURFACE}; border-radius: 10px; padding: 6px 8px; }}
.ds-nav-step .i {{ font-size: 1.1rem; line-height: 1.2; }}
.ds-nav-step b {{ display: block; font-size: .8rem; color: {INK}; }}
.ds-nav-step span {{ display: block; font-size: .7rem; color: {MUTED}; line-height: 1.2; }}
.ds-glossary {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 10px; margin: 6px 0 10px; }}
.ds-gloss {{ background: #ffffff; border: 1px solid rgba(11,11,11,0.08); border-radius: 12px; padding: 10px 12px; }}
.ds-gloss b {{ display: block; color: {INK}; font-size: .9rem; margin-bottom: 3px; }}
.ds-gloss span {{ color: {INK_2}; font-size: .8rem; line-height: 1.35; }}
.ds-roles > div {{ background: {SURFACE}; border-radius: 10px; padding: 8px 10px; font-size: .84rem; color: {INK_2}; }}
.ds-roles b {{ color: {INK}; display: block; }}
.ds-section {{ font-size: 1.12rem; font-weight: 750; color: {INK}; margin: 14px 0 4px; padding-bottom: 4px;
  border-bottom: 2px solid #e8f1fc; }}
.ds-settings {{ border: 1px solid rgba(42,120,214,0.22) !important; border-radius: 16px !important; background: #fbfdff !important; }}
.ds-settings .ds-note, .ds-settings .prose {{ padding: 8px 14px !important; }}
@media (max-width: 1100px) {{ .ds-kpis, .ds-hero .ds-kpis {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
  .ds-nav-row {{ grid-template-columns: repeat(3, minmax(0,1fr)); }} .ds-glossary {{ grid-template-columns: 1fr 1fr; }}
  .ds-grid3, .ds-grid4 {{ grid-template-columns: 1fr 1fr; }} }}
@media (max-width: 700px) {{ .ds-kpis, .ds-hero .ds-kpis, .ds-grid2, .ds-grid3, .ds-grid4, .ds-glossary, .ds-nav-row {{ grid-template-columns: 1fr; }}
  .ds-hero h1 {{ font-size: 1.6rem; }} }}
"""
