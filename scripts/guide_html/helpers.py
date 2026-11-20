"""HTML building blocks and image embedding for the guide (everything ends up in one self-contained file)."""
from __future__ import annotations

import base64
import io
from html import escape
from pathlib import Path

import numpy as np
from PIL import Image

TOC: list[tuple[int, str | None, str]] = []  # (level 0 = part, 1 = chapter, 2 = sub-section; id; title)


def e(value) -> str:
    return escape(str(value), quote=True)


def part(title: str) -> str:
    TOC.append((0, None, title))
    return f'<div class="part-title">{e(title)}</div>'


def chapter(cid: str, num, title: str, body: str, lede: str | None = None) -> str:
    TOC.append((1, cid, f"{num}. {title}"))
    lede_html = f'<p class="lede">{lede}</p>' if lede else ""
    return (f'<section class="chapter" id="{cid}"><h2><span class="n">{num}</span>{e(title)}'
            f'<a class="anchor" href="#{cid}" aria-label="link to this section">#</a></h2>{lede_html}{body}</section>')


def sub(cid: str, num: str, title: str) -> str:
    TOC.append((2, cid, f"{num} {title}"))
    return (f'<h3 id="{cid}"><span class="n">{num}</span>{e(title)}'
            f'<a class="anchor" href="#{cid}" aria-label="link to this section">#</a></h3>')


def callout(kind: str, title: str, body: str) -> str:
    """kind: what (what is happening) · why · tip · warn · code (in the code) · key (key point)."""
    return f'<div class="callout {kind}"><div class="t">{title}</div>{body}</div>'


def steps(items) -> str:
    """[(heading, body_html, optional time label), ...] -> numbered step cards."""
    out = []
    for item in items:
        head, body = item[0], item[1]
        tm = f'<span class="tm">{item[2]}</span>' if len(item) > 2 and item[2] else ""
        out.append(f'<li>{tm}<b class="h">{head}</b>{body}</li>')
    return f'<ol class="steps">{"".join(out)}</ol>'


def table(header, rows, num_cols=(), caption: str | None = None, hl_rows=()) -> str:
    th = "".join(f'<th class="{"num" if i in num_cols else ""}">{h}</th>' for i, h in enumerate(header))
    body = []
    for r, row in enumerate(rows):
        tds = "".join(f'<td class="{"num" if i in num_cols else ""}">{c}</td>' for i, c in enumerate(row))
        body.append(f'<tr class="{"hl" if r in hl_rows else ""}">{tds}</tr>')
    cap = f"<caption>{caption}</caption>" if caption else ""
    return (f'<div class="tbl-wrap"><table class="tbl">{cap}<thead><tr>{th}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')


def kpis(items) -> str:
    tiles = "".join(f'<div class="kpi"><div class="v">{v}</div><div class="l">{label}</div><div class="s">{s}</div></div>'
                    for v, label, s in items)
    return f'<div class="kpis">{tiles}</div>'


def code(text: str, lang: str = "") -> str:
    return f'<pre><code class="lang-{lang}">{e(text.strip())}</code></pre>'


def details(summary: str, body: str, open_: bool = False) -> str:
    return f'<details class="deep"{" open" if open_ else ""}><summary>{summary}</summary><div>{body}</div></details>'


def formula(text: str) -> str:
    return f'<div class="formula">{e(text.strip())}</div>'


def dec(decision: str) -> str:
    icon = {"PASS": "✓", "CHECK": "⚠", "REJECT": "✕"}[decision]
    return f'<span class="dec {decision}">{icon} {decision}</span>'


def chips(items) -> str:
    return '<div class="chips">' + "".join(f'<span class="chip">{c}</span>' for c in items) + "</div>"


def zones(threshold: float, band: float, label: str = "") -> str:
    lo, hi = threshold - band, threshold + band
    aria = f"PASS below {lo:g}, CHECK from {lo:g} to {hi:g}, REJECT from {hi:g} on the 0 to 100 score scale"
    return (f'<div class="zones" style="grid-template-columns:{lo:g}fr {2 * band:g}fr {100 - hi:g}fr" role="img" '
            f'aria-label="{e(aria)}"><div class="p">✓ PASS &lt; {lo:g}</div><div class="c">⚠ CHECK</div>'
            f'<div class="r">✕ REJECT ≥ {hi:g}</div></div>'
            f'<div class="zone-axis"><span>0</span><span>{label}</span><span>100</span></div>')


# ------------------------------------------------------------------------------------------------------- images
def _encode(im: Image.Image, quality: int) -> str:
    buf = io.BytesIO()
    im.save(buf, "WEBP", quality=quality, method=6)
    return "data:image/webp;base64," + base64.b64encode(buf.getvalue()).decode()


def img_uri(path: str | Path, max_w: int = 1400, quality: int = 80) -> str:
    im = Image.open(path).convert("RGB")
    if im.width > max_w:
        im = im.resize((max_w, round(im.height * max_w / im.width)), Image.LANCZOS)
    return _encode(im, quality)


def thumb_uri(source, size: int = 220, quality: int = 78) -> str:
    """Square centre-cropped thumbnail of a path or an RGB array."""
    im = Image.fromarray(np.asarray(source)) if isinstance(source, np.ndarray) else Image.open(source)
    im = im.convert("RGB")
    side = min(im.size)
    left, top = (im.width - side) // 2, (im.height - side) // 2
    im = im.crop((left, top, left + side, top + side)).resize((size, size), Image.LANCZOS)
    return _encode(im, quality)


def fig(uri: str, caption: str, alt: str, cls: str = "") -> str:
    return (f'<figure class="{cls}"><img src="{uri}" alt="{e(alt)}">'
            f'<figcaption>{caption}</figcaption></figure>')


def thumbs(items) -> str:
    """[(uri, caption), ...] -> a responsive grid of small images."""
    cells = "".join(f'<figure><img src="{u}" alt="{e(c)}"><figcaption>{c}</figcaption></figure>'
                    for u, c in items)
    return f'<div class="thumbs">{cells}</div>'


def pct(x, digits: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * float(x):.{digits}f}%"


def f3(x) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{float(x):.3f}"
