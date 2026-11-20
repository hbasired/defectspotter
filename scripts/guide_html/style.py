"""CSS and JavaScript of the self-contained HTML guide (light + dark theme, sidebar, lightbox, search)."""

CSS = r"""
:root {
  --bg: #f5f6f8; --surface: #ffffff; --surface-2: #f3f3f1; --surface-3: #ebebe7;
  --ink: #0b0b0b; --ink-2: #3d3d39; --muted: #64635d; --line: #e1e0d9; --line-strong: #c3c2b7;
  --accent: #2a78d6; --accent-ink: #1b5fae; --accent-bg: #eaf2fc;
  --violet: #6250d6; --violet-bg: #efedfb; --green: #13865d; --green-bg: #e6f6ef;
  --amber-bg: #fff6dc; --amber-ink: #7a5200; --red-bg: #fdeceb; --red-ink: #9b1c1c;
  --pass: #0ca30c; --check: #e3a008; --reject: #d03b3b;
  --pass-bg: #eaf7ea; --check-bg: #fff6dc; --reject-bg: #fdeceb;
  --pass-ink: #006300; --check-ink: #7a5200; --reject-ink: #9b1c1c;
  --code-bg: #f4f4f1; --shadow: 0 1px 2px rgba(0,0,0,.05), 0 4px 16px rgba(0,0,0,.05);
  /* diagram tokens */
  --d-proc: #ffffff; --d-proc-line: #a9a89f;
  --d-data: #eaf2fc; --d-data-line: #2a78d6;
  --d-model: #efedfb; --d-model-line: #6250d6;
  --d-gate: #fff6dc; --d-gate-line: #c98a00;
  --d-human: #e6f6ef; --d-human-line: #13865d;
  --d-alarm: #fdeceb; --d-alarm-line: #d03b3b;
  --d-store: #f3f3f1; --d-store-line: #7d7c74;
  --d-lane: #f7f7f5; --d-edge: #6f6e68;
  color-scheme: light;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --bg: #0e1013; --surface: #16191e; --surface-2: #1d2127; --surface-3: #252a31;
    --ink: #ecebe7; --ink-2: #cfcec8; --muted: #a09f98; --line: #2a2e35; --line-strong: #3d424b;
    --accent: #5ea1ec; --accent-ink: #93c1f4; --accent-bg: #15263b;
    --violet: #a497f2; --violet-bg: #221f3d; --green: #4cc895; --green-bg: #11291f;
    --amber-bg: #2f2610; --amber-ink: #f3c860; --red-bg: #351717; --red-ink: #f59b9b;
    --pass: #3fcf3f; --check: #f5b82e; --reject: #f06464;
    --pass-bg: #112a13; --check-bg: #2f2610; --reject-bg: #351717;
    --pass-ink: #8fe08f; --check-ink: #f3c860; --reject-ink: #f59b9b;
    --code-bg: #1b1f25; --shadow: 0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
    --d-proc: #1d2127; --d-proc-line: #5d626b;
    --d-data: #15263b; --d-data-line: #5ea1ec;
    --d-model: #221f3d; --d-model-line: #a497f2;
    --d-gate: #2f2610; --d-gate-line: #e0a52a;
    --d-human: #11291f; --d-human-line: #4cc895;
    --d-alarm: #351717; --d-alarm-line: #f06464;
    --d-store: #252a31; --d-store-line: #8b8f97;
    --d-lane: #13161a; --d-edge: #9a9ea6;
    color-scheme: dark;
  }
}
:root[data-theme="dark"] {
  --bg: #0e1013; --surface: #16191e; --surface-2: #1d2127; --surface-3: #252a31;
  --ink: #ecebe7; --ink-2: #cfcec8; --muted: #a09f98; --line: #2a2e35; --line-strong: #3d424b;
  --accent: #5ea1ec; --accent-ink: #93c1f4; --accent-bg: #15263b;
  --violet: #a497f2; --violet-bg: #221f3d; --green: #4cc895; --green-bg: #11291f;
  --amber-bg: #2f2610; --amber-ink: #f3c860; --red-bg: #351717; --red-ink: #f59b9b;
  --pass: #3fcf3f; --check: #f5b82e; --reject: #f06464;
  --pass-bg: #112a13; --check-bg: #2f2610; --reject-bg: #351717;
  --pass-ink: #8fe08f; --check-ink: #f3c860; --reject-ink: #f59b9b;
  --code-bg: #1b1f25; --shadow: 0 1px 2px rgba(0,0,0,.4), 0 4px 16px rgba(0,0,0,.3);
  --d-proc: #1d2127; --d-proc-line: #5d626b;
  --d-data: #15263b; --d-data-line: #5ea1ec;
  --d-model: #221f3d; --d-model-line: #a497f2;
  --d-gate: #2f2610; --d-gate-line: #e0a52a;
  --d-human: #11291f; --d-human-line: #4cc895;
  --d-alarm: #351717; --d-alarm-line: #f06464;
  --d-store: #252a31; --d-store-line: #8b8f97;
  --d-lane: #13161a; --d-edge: #9a9ea6;
  color-scheme: dark;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; scroll-padding-top: 72px; -webkit-text-size-adjust: 100%; }
body { margin: 0; background: var(--bg); color: var(--ink); font: 16px/1.62 system-ui, -apple-system, "Segoe UI", Roboto,
  "Helvetica Neue", Arial, sans-serif; }
a { color: var(--accent-ink); text-decoration-thickness: 1px; text-underline-offset: 2px; }
code, pre, kbd, .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, "Liberation Mono", monospace; }
code { background: var(--code-bg); border: 1px solid var(--line); border-radius: 5px; padding: .05em .35em; font-size: .88em;
  overflow-wrap: anywhere; }
pre { background: var(--code-bg); border: 1px solid var(--line); border-radius: 10px; padding: 14px 16px; overflow-x: auto;
  font-size: 13.5px; line-height: 1.5; position: relative; margin: 14px 0; }
pre code { background: none; border: 0; padding: 0; font-size: inherit; overflow-wrap: normal; }
.copy { position: absolute; top: 8px; right: 8px; font: 12px system-ui, sans-serif; padding: 3px 9px; border-radius: 6px;
  border: 1px solid var(--line-strong); background: var(--surface); color: var(--ink-2); cursor: pointer; }
.copy:hover { color: var(--ink); border-color: var(--accent); }

/* top bar */
.topbar { position: sticky; top: 0; z-index: 40; display: flex; align-items: center; gap: 12px; height: 56px; padding: 0 16px;
  background: color-mix(in srgb, var(--surface) 92%, transparent); backdrop-filter: blur(8px); border-bottom: 1px solid var(--line); }
.topbar .brand { font-weight: 700; letter-spacing: -.01em; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
.topbar .brand span { color: var(--muted); font-weight: 500; }
.topbar .spacer { flex: 1; }
.btn { display: inline-flex; align-items: center; gap: 6px; padding: 6px 12px; border-radius: 8px; border: 1px solid var(--line-strong);
  background: var(--surface); color: var(--ink); font: 500 14px system-ui, sans-serif; cursor: pointer; text-decoration: none; white-space: nowrap; }
.btn:hover { border-color: var(--accent); }
.btn.primary { background: var(--accent); border-color: var(--accent); color: #fff; }
:root[data-theme="dark"] .btn.primary { color: #0b1420; }
#tocbtn { display: none; }
.progress { position: absolute; left: 0; bottom: -1px; height: 2px; background: var(--accent); width: 0; }

/* layout */
.layout { display: grid; grid-template-columns: 290px minmax(0, 1fr); max-width: 1440px; margin: 0 auto; }
nav#toc { position: sticky; top: 56px; height: calc(100vh - 56px); overflow-y: auto; padding: 18px 12px 40px 18px;
  border-right: 1px solid var(--line); font-size: 14px; }
nav#toc input { width: 100%; padding: 8px 10px; border-radius: 8px; border: 1px solid var(--line-strong); background: var(--surface);
  color: var(--ink); font: 14px system-ui, sans-serif; margin-bottom: 6px; }
nav#toc .hint { color: var(--muted); font-size: 12px; margin: 0 0 10px 2px; min-height: 1em; }
nav#toc .part { margin: 16px 0 4px 2px; font-size: 11.5px; font-weight: 700; letter-spacing: .08em; text-transform: uppercase; color: var(--muted); }
nav#toc a { display: block; padding: 4px 8px; border-radius: 6px; color: var(--ink-2); text-decoration: none; line-height: 1.35; }
nav#toc a.sub { padding-left: 26px; font-size: 13px; color: var(--muted); }
nav#toc a:hover { background: var(--surface-2); color: var(--ink); }
nav#toc a.active { background: var(--accent-bg); color: var(--accent-ink); font-weight: 600; }
nav#toc a.hide { display: none; }
main { padding: 8px 40px 80px; min-width: 0; }
.wrap { max-width: 980px; margin: 0 auto; }

/* hero */
.hero { margin: 26px 0 8px; padding: 30px 32px; border-radius: 18px; background: var(--surface); border: 1px solid var(--line); box-shadow: var(--shadow); }
.hero .eyebrow { font-size: 13px; font-weight: 600; letter-spacing: .06em; text-transform: uppercase; color: var(--accent-ink); }
.hero h1 { font-size: 40px; line-height: 1.12; margin: 8px 0 10px; letter-spacing: -.02em; }
.hero p.lead { font-size: 18.5px; color: var(--ink-2); margin: 0 0 18px; max-width: 760px; }
.hero .links { display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 22px; }
.meta { color: var(--muted); font-size: 13.5px; margin-top: 14px; }

/* KPI tiles */
.kpis { display: grid; grid-template-columns: repeat(auto-fill, minmax(168px, 1fr)); gap: 10px; margin: 14px 0; }
.kpi { background: var(--surface-2); border: 1px solid var(--line); border-radius: 12px; padding: 12px 14px; }
.kpi .v { font-size: 26px; font-weight: 700; letter-spacing: -.02em; line-height: 1.15; color: var(--ink); }
.kpi .l { font-size: 13.5px; font-weight: 600; color: var(--ink-2); margin-top: 2px; }
.kpi .s { font-size: 12.5px; color: var(--muted); margin-top: 3px; line-height: 1.35; }

/* sections */
section.chapter { margin-top: 30px; padding-top: 6px; }
.part-title { margin: 56px 0 0; padding: 14px 0 0; border-top: 2px solid var(--line-strong); font-size: 13px; font-weight: 700;
  letter-spacing: .1em; text-transform: uppercase; color: var(--muted); }
h2 { font-size: 29px; line-height: 1.2; letter-spacing: -.015em; margin: 18px 0 10px; }
h2 .n { color: var(--accent-ink); margin-right: 10px; }
h3 { font-size: 21px; line-height: 1.3; margin: 32px 0 8px; letter-spacing: -.01em; }
h3 .n { color: var(--muted); margin-right: 8px; font-weight: 600; }
h4 { font-size: 16.5px; margin: 22px 0 6px; }
p, li { color: var(--ink); }
.lede { font-size: 17.5px; color: var(--ink-2); }
.small { font-size: 13.5px; color: var(--muted); }
ul, ol { padding-left: 1.35em; }
li { margin: 4px 0; }
hr { border: 0; border-top: 1px solid var(--line); margin: 28px 0; }
.anchor { opacity: 0; margin-left: 8px; font-size: .7em; text-decoration: none; color: var(--muted); }
h2:hover .anchor, h3:hover .anchor { opacity: 1; }

/* callouts */
.callout { border: 1px solid var(--line); border-left: 4px solid var(--line-strong); background: var(--surface); border-radius: 10px;
  padding: 12px 16px; margin: 16px 0; }
.callout > .t { font-weight: 700; font-size: 14.5px; margin-bottom: 4px; display: flex; gap: 8px; align-items: center; }
.callout > *:last-child { margin-bottom: 0; }
.callout p { margin: 6px 0; }
.callout.what { border-left-color: var(--accent); background: var(--accent-bg); }
.callout.why { border-left-color: var(--violet); background: var(--violet-bg); }
.callout.tip { border-left-color: var(--green); background: var(--green-bg); }
.callout.warn { border-left-color: var(--check); background: var(--amber-bg); }
.callout.code { border-left-color: var(--line-strong); background: var(--surface-2); }
.callout.key { border-left-color: var(--reject); background: var(--red-bg); }

/* steps */
ol.steps { list-style: none; padding: 0; counter-reset: s; margin: 14px 0; }
ol.steps > li { counter-increment: s; position: relative; padding: 10px 14px 10px 54px; margin: 8px 0; background: var(--surface);
  border: 1px solid var(--line); border-radius: 10px; }
ol.steps > li::before { content: counter(s); position: absolute; left: 14px; top: 10px; width: 28px; height: 28px; border-radius: 50%;
  background: var(--accent); color: #fff; font-weight: 700; font-size: 14px; display: grid; place-items: center; }
:root[data-theme="dark"] ol.steps > li::before { color: #0b1420; }
ol.steps > li b.h { display: block; margin-bottom: 2px; }
ol.steps > li .tm { float: right; font-size: 12.5px; color: var(--muted); margin-left: 8px; }

/* tables */
.tbl-wrap { overflow-x: auto; margin: 14px 0; border: 1px solid var(--line); border-radius: 10px; background: var(--surface); }
table.tbl { border-collapse: collapse; width: 100%; font-size: 14px; }
table.tbl th, table.tbl td { padding: 7px 11px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
table.tbl th { background: var(--surface-2); font-weight: 600; color: var(--ink-2); white-space: nowrap; position: sticky; top: 0; }
table.tbl tr:last-child td { border-bottom: 0; }
table.tbl td.num, table.tbl th.num { text-align: right; font-variant-numeric: tabular-nums; white-space: nowrap; }
table.tbl tr.hl td { background: var(--accent-bg); font-weight: 600; }
table.tbl caption { caption-side: bottom; text-align: left; padding: 8px 12px; color: var(--muted); font-size: 13px; }
.tbl-note { font-size: 13px; color: var(--muted); margin: -6px 0 14px; }

/* decisions */
.dec { display: inline-flex; align-items: center; gap: 4px; font-weight: 700; font-size: .86em; padding: 1px 8px; border-radius: 999px; border: 1px solid; white-space: nowrap; }
.dec.PASS { color: var(--pass-ink); background: var(--pass-bg); border-color: var(--pass); }
.dec.CHECK { color: var(--check-ink); background: var(--check-bg); border-color: var(--check); }
.dec.REJECT { color: var(--reject-ink); background: var(--reject-bg); border-color: var(--reject); }
.zones { display: grid; grid-template-columns: var(--z-pass) var(--z-check) 1fr; border-radius: 10px; overflow: hidden; border: 1px solid var(--line);
  margin: 12px 0 4px; font-size: 13.5px; font-weight: 600; }
.zones > div { padding: 10px 12px; }
.zones .p { background: var(--pass-bg); color: var(--pass-ink); }
.zones .c { background: var(--check-bg); color: var(--check-ink); text-align: center; border-left: 2px solid var(--check); border-right: 2px solid var(--check); }
.zones .r { background: var(--reject-bg); color: var(--reject-ink); text-align: right; }
.zone-axis { display: flex; justify-content: space-between; font-size: 12px; color: var(--muted); font-variant-numeric: tabular-nums; }

/* figures */
figure { margin: 18px 0; }
figure img { display: block; max-width: 100%; height: auto; border-radius: 10px; border: 1px solid var(--line); background: var(--surface); cursor: zoom-in; }
figure.shot img { box-shadow: var(--shadow); }
figcaption { font-size: 13.5px; color: var(--muted); margin-top: 8px; line-height: 1.45; }
figcaption b { color: var(--ink-2); }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
.grid3 { display: grid; grid-template-columns: repeat(3, 1fr); gap: 12px; }
.grid4 { display: grid; grid-template-columns: repeat(4, 1fr); gap: 12px; }
.grid2 > *, .grid3 > *, .grid4 > *, .cols > *, .kpis > * { min-width: 0; }
.thumbs { display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 10px; margin: 12px 0; }
.thumbs figure { margin: 0; }
.thumbs img { aspect-ratio: 1 / 1; object-fit: cover; width: 100%; }
.thumbs figcaption { font-size: 12px; margin-top: 4px; }

/* diagrams */
.diagram { margin: 18px 0; padding: 12px; background: var(--surface); border: 1px solid var(--line); border-radius: 14px; overflow-x: auto; }
.diagram svg { display: block; width: 100%; height: auto; min-width: 760px; }
.diagram .cap { font-size: 13.5px; color: var(--muted); margin: 8px 4px 0; }
.legend { display: flex; flex-wrap: wrap; gap: 6px 14px; font-size: 12.5px; color: var(--ink-2); margin: 8px 4px 2px; }
.legend i { display: inline-block; width: 14px; height: 11px; border-radius: 3px; border: 1.5px solid; margin-right: 5px; vertical-align: -1px; }
svg .n-proc { fill: var(--d-proc); stroke: var(--d-proc-line); }
svg .n-data { fill: var(--d-data); stroke: var(--d-data-line); }
svg .n-model { fill: var(--d-model); stroke: var(--d-model-line); }
svg .n-gate { fill: var(--d-gate); stroke: var(--d-gate-line); }
svg .n-human { fill: var(--d-human); stroke: var(--d-human-line); }
svg .n-alarm { fill: var(--d-alarm); stroke: var(--d-alarm-line); }
svg .n-store { fill: var(--d-store); stroke: var(--d-store-line); }
svg .n-ghost { fill: transparent; stroke: var(--d-proc-line); stroke-dasharray: 5 4; }
svg .shape { stroke-width: 1.6; }
svg .t1 { fill: var(--ink); font: 600 14px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
svg .t2 { fill: var(--ink-2); font: 12.5px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif; }
svg .mono { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
svg .edge { fill: none; stroke: var(--d-edge); stroke-width: 1.7; }
svg .edge.dash { stroke-dasharray: 6 5; }
svg .ah { fill: var(--d-edge); }
svg .el { fill: var(--ink-2); font: 600 12px system-ui, sans-serif; paint-order: stroke; stroke: var(--surface); stroke-width: 5px; stroke-linejoin: round; }
svg .lane { fill: var(--d-lane); stroke: var(--line); }
svg .lane-t { fill: var(--muted); font: 700 12.5px system-ui, sans-serif; letter-spacing: .02em; }
svg .badge { fill: var(--accent); }
svg .badge-t { fill: #fff; font: 700 11.5px system-ui, sans-serif; }
:root[data-theme="dark"] svg .badge-t { fill: #0b1420; }
svg .ann { fill: var(--muted); font: italic 12px system-ui, sans-serif; }

/* details */
details.deep { border: 1px solid var(--line); border-radius: 10px; background: var(--surface); margin: 14px 0; }
details.deep > summary { cursor: pointer; padding: 10px 14px; font-weight: 600; color: var(--ink-2); list-style: none; }
details.deep > summary::-webkit-details-marker { display: none; }
details.deep > summary::before { content: "▸"; display: inline-block; width: 1.1em; color: var(--accent-ink); transition: transform .15s; }
details.deep[open] > summary::before { transform: rotate(90deg); }
details.deep > div { padding: 0 16px 12px; }

.formula { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; background: var(--surface-2); border: 1px solid var(--line);
  border-radius: 10px; padding: 12px 16px; margin: 12px 0; overflow-x: auto; white-space: pre; font-size: 14px; line-height: 1.55; }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin: 8px 0; }
.chip { font-size: 12.5px; padding: 2px 9px; border-radius: 999px; background: var(--surface-2); border: 1px solid var(--line); color: var(--ink-2); }
.cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
.card { background: var(--surface); border: 1px solid var(--line); border-radius: 12px; padding: 14px 16px; }
.card h4 { margin-top: 0; }
dl.gloss { display: grid; grid-template-columns: minmax(150px, 220px) 1fr; gap: 6px 18px; }
dl.gloss dt { font-weight: 700; }
dl.gloss dd { margin: 0; color: var(--ink-2); }
.tree { font-size: 13.5px; line-height: 1.5; }
.backtop { position: fixed; right: 18px; bottom: 18px; z-index: 30; display: none; }

/* lightbox */
#lightbox { position: fixed; inset: 0; z-index: 100; background: rgba(0,0,0,.84); display: none; align-items: center; justify-content: center;
  padding: 24px; cursor: zoom-out; }
#lightbox img { max-width: 100%; max-height: 100%; border-radius: 8px; box-shadow: 0 10px 40px rgba(0,0,0,.5); }
#lightbox.on { display: flex; }

@media (max-width: 1080px) {
  .layout { grid-template-columns: 1fr; }
  nav#toc { position: fixed; left: 0; top: 56px; bottom: 0; width: min(320px, 88vw); height: auto; z-index: 50; background: var(--surface);
    transform: translateX(-102%); transition: transform .2s; box-shadow: var(--shadow); }
  nav#toc.open { transform: none; }
  #tocbtn { display: inline-flex; }
  main { padding: 4px 16px 60px; }
}
@media (max-width: 720px) {
  .hero { padding: 20px 16px; }
  .hero h1 { font-size: 30px; }
  h2 { font-size: 24px; }
  h3 { font-size: 19px; }
  .grid2, .grid3, .cols { grid-template-columns: 1fr; }
  .grid4 { grid-template-columns: 1fr 1fr; }
  dl.gloss { grid-template-columns: 1fr; }
  dl.gloss dd { margin-bottom: 8px; }
  .topbar .brand span { display: none; }
  ol.steps > li .tm { float: none; display: block; margin: 0; }
}
@media print {
  .topbar, nav#toc, .backtop, .copy { display: none !important; }
  .layout { display: block; }
  main { padding: 0; }
  details.deep { break-inside: avoid; }
  details.deep > div { display: block !important; }
  section.chapter { break-before: auto; }
  .diagram, figure, .tbl-wrap { break-inside: avoid; }
}
"""

JS = r"""
(function () {
  var root = document.documentElement;
  function stored() { try { return localStorage.getItem('ds-guide-theme'); } catch (e) { return null; } }
  function store(v) { try { localStorage.setItem('ds-guide-theme', v); } catch (e) {} }
  var s = stored(); if (s === 'light' || s === 'dark') root.setAttribute('data-theme', s);
  function current() {
    var t = root.getAttribute('data-theme');
    if (t) return t;
    return window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  }
  function label() { var b = document.getElementById('themebtn'); if (b) b.textContent = current() === 'dark' ? '☀ Light' : '☾ Dark'; }
  document.addEventListener('DOMContentLoaded', function () {
    label();
    document.getElementById('themebtn').addEventListener('click', function () {
      var next = current() === 'dark' ? 'light' : 'dark'; root.setAttribute('data-theme', next); store(next); label();
    });
    var toc = document.getElementById('toc');
    document.getElementById('tocbtn').addEventListener('click', function () { toc.classList.toggle('open'); });
    toc.addEventListener('click', function (e) { if (e.target.tagName === 'A' && window.innerWidth <= 1080) toc.classList.remove('open'); });
    // copy buttons
    document.querySelectorAll('pre').forEach(function (pre) {
      var b = document.createElement('button'); b.className = 'copy'; b.type = 'button'; b.textContent = 'Copy';
      b.addEventListener('click', function () {
        var text = pre.querySelector('code') ? pre.querySelector('code').innerText : pre.innerText;
        var done = function () { b.textContent = 'Copied'; setTimeout(function () { b.textContent = 'Copy'; }, 1400); };
        if (navigator.clipboard) navigator.clipboard.writeText(text).then(done, function () {}); else done();
      });
      pre.appendChild(b);
    });
    // lightbox
    var lb = document.getElementById('lightbox'), lbi = lb.querySelector('img');
    document.querySelectorAll('figure img').forEach(function (img) {
      img.addEventListener('click', function () { lbi.src = img.src; lbi.alt = img.alt; lb.classList.add('on'); });
    });
    lb.addEventListener('click', function () { lb.classList.remove('on'); });
    document.addEventListener('keydown', function (e) { if (e.key === 'Escape') lb.classList.remove('on'); });
    // active section + progress
    var links = Array.prototype.slice.call(toc.querySelectorAll('a[href^="#"]'));
    var byId = {}; links.forEach(function (a) { byId[a.getAttribute('href').slice(1)] = a; });
    var targets = links.map(function (a) { return document.getElementById(a.getAttribute('href').slice(1)); }).filter(Boolean);
    var bar = document.querySelector('.progress'), back = document.querySelector('.backtop');
    function onScroll() {
      var y = window.scrollY + 90, cur = null;
      for (var i = 0; i < targets.length; i++) { if (targets[i].offsetTop <= y) cur = targets[i]; else break; }
      links.forEach(function (a) { a.classList.remove('active'); });
      if (cur && byId[cur.id]) { byId[cur.id].classList.add('active'); }
      var h = document.documentElement.scrollHeight - window.innerHeight;
      bar.style.width = (h > 0 ? 100 * window.scrollY / h : 0) + '%';
      back.style.display = window.scrollY > 900 ? 'inline-flex' : 'none';
    }
    window.addEventListener('scroll', onScroll, { passive: true }); onScroll();
    // search: filter the table of contents by section text
    var q = document.getElementById('tocsearch'), hint = document.getElementById('tochint');
    var texts = {}; targets.forEach(function (t) {
      if (t.tagName === 'SECTION') { texts[t.id] = t.textContent.toLowerCase(); return; }
      var s = t.textContent, nx = t.nextElementSibling;
      while (nx && nx.tagName !== 'H3' && nx.tagName !== 'H2') { s += ' ' + nx.textContent; nx = nx.nextElementSibling; }
      texts[t.id] = s.toLowerCase();
    });
    q.addEventListener('input', function () {
      var v = q.value.trim().toLowerCase(), n = 0;
      links.forEach(function (a) {
        var id = a.getAttribute('href').slice(1);
        var hit = !v || (texts[id] || '').indexOf(v) >= 0 || a.textContent.toLowerCase().indexOf(v) >= 0;
        a.classList.toggle('hide', !hit); if (hit && v) n++;
      });
      hint.textContent = v ? (n ? n + ' section(s) mention "' + q.value.trim() + '"' : 'no section mentions "' + q.value.trim() + '"') : '';
    });
  });
})();
"""
