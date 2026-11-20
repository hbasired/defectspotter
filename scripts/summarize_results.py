"""Headline result tables (Markdown) from reports/; ``--write`` puts them into README.md and PROJECT_HANDOFF.md."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
LABELS = {"patchcore": "PatchCore", "fvae": "Feature VAE", "vae": "Pixel VAE", "mean": "Mean baseline", "ensemble": "Ensemble"}


def table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def recipe_table() -> str | None:
    path = R / "hpo_patchcore.csv"
    if not path.exists():
        return None
    hpo = pd.read_csv(path)
    if "selected" not in hpo:
        return None
    sel = hpo[hpo.selected.astype(str).str.lower() == "true"].copy()
    rows = pd.DataFrame({
        "product": sel.category,
        "input": sel.image_size.astype(int).astype(str) + " px",
        "pose": (sel["align"].astype(str).str.lower() == "true").map({True: "normalised", False: "as captured"})
        if "align" in sel else "as captured",
        "descriptors": sel.pool_size.map(lambda v: "3×3 average" if int(v) == 3 else "raw"),
        "good parts in bank": sel.bank_rotations.map(lambda v: "as captured" if int(v) == 1 else f"{int(v)} orientations"),
        "bank": sel.bank_vectors.astype(int).map("{:,}".format) + " vectors (" + sel.bank_mb.map("{:.1f}".format) + " MB)",
        "score": sel.apply(lambda r: "max patch" if r.score_mode == "max" else f"top {100 * r.topk_fraction:.0f}% of map", axis=1),
        "label-free objective": sel.objective.map("{:.4f}".format)})
    return "### Per-product PatchCore recipe (chosen without real defects)\n" + table(rows)


def build() -> str:
    metrics = pd.read_csv(R / "metrics.csv")
    out = []
    recipe = recipe_table()
    if recipe:
        out.append(recipe)
    m = metrics.copy()
    m["image AUROC [95% CI]"] = m.apply(lambda r: f"{r.image_auroc:.3f} [{r.image_auroc_ci_low:.3f}–{r.image_auroc_ci_high:.3f}]", axis=1)
    m["pixel AUROC"] = m.pixel_auroc.map("{:.3f}".format)
    m["AUPRO@30%"] = m.aupro_30.map("{:.3f}".format)
    m["method"] = m.method.map(LABELS)
    out.append("### Detection and localisation (MVTec test images)\n" + table(m[["category", "method", "image AUROC [95% CI]", "pixel AUROC", "AUPRO@30%"]]))
    macro = metrics.groupby("method")[["image_auroc", "pixel_auroc", "aupro_30"]].mean().round(3).reset_index()
    macro["method"] = macro.method.map(LABELS)
    out.append("### Mean over the three products\n" + table(macro))
    req_path = R / "requalification.csv"
    if req_path.exists():
        rq = pd.read_csv(req_path)
        rq = rq.assign(**{"lot-shift test": rq.apply(lambda r: f"p = {r.p_value:.3g} → " + ("re-qualified" if r.lot_shift else "no change"), axis=1),
                          "good parts → CHECK": rq.apply(lambda r: f"{100 * r.base_good_check_rate:.1f}% → {100 * r.requal_good_check_rate:.1f}%" if r.lot_shift else "—", axis=1),
                          "escapes": rq.apply(lambda r: f"{100 * r.base_escape_rate:.1f}% → {100 * r.requal_escape_rate:.1f}%" if r.lot_shift else "—", axis=1),
                          "false rejects": rq.apply(lambda r: f"{100 * r.base_false_reject_rate:.1f}% → {100 * r.requal_false_reject_rate:.1f}%" if r.lot_shift else "—", axis=1)})
        out.append("### New production lot: label-free lot-shift test and re-qualification (operating split)\n" + table(
            rq[["category", "n_good_current_lot", "lot-shift test", "good parts → CHECK", "escapes", "false rejects"]]))
    ops = pd.read_csv(R / "operating_metrics.csv")
    o = ops[ops.policy.isin(["synthetic_cost", "real_oracle_cost"])].copy()
    o = o[o.method.isin(["patchcore", "fvae", "ensemble"])]
    o["method"] = o.method.map(LABELS)
    o = o.assign(**{"threshold ± band": o.threshold.map("{:.1f}".format) + " ± " + o.band.map("{:.1f}".format),
                    "false rejects": (100 * o.false_reject_rate).map("{:.1f}%".format),
                    "escapes": (100 * o.escape_rate).map("{:.1f}%".format),
                    "CHECK load": (100 * o.review_rate).map("{:.1f}%".format),
                    "cost / 1,000 parts": o.expected_cost_per_1000.map("{:.0f}".format)})
    out.append("### Operating point on the held-out operating split (1% prevalence cost model)\n" + table(
        o[["category", "method", "policy", "threshold ± band", "false rejects", "escapes", "CHECK load", "cost / 1,000 parts"]]))
    thr = pd.read_csv(R / "threshold_comparison.csv")
    cols = ["category", "method", "synthetic_threshold", "real_threshold", "gap_synthetic_vs_real"]
    if "diffusion_threshold" in thr:
        cols += ["diffusion_threshold", "gap_diffusion_vs_real"]
    t = thr[thr.method.isin(["patchcore", "fvae"])][cols].round(1)
    t["method"] = t.method.map(LABELS)
    out.append("### Threshold chosen with synthetic defects vs real defects (0–100 scale)\n" + table(t))
    lat = pd.read_csv(R / "latency.csv")
    lat = lat[["category", "method", "latency_ms_median", "latency_ms_p95"]].round(0)
    lat["method"] = lat.method.map(LABELS)
    out.append("### Latency per part (CPU, ONNX Runtime, end-to-end)\n" + table(lat))
    if (R / "few_shot.csv").exists():
        few = pd.read_csv(R / "few_shot.csv").groupby("shots")[["image_auroc", "pixel_auroc"]].mean().round(3).reset_index()
        out.append("### Few-shot onboarding (metal nut)\n" + table(few))
    cost = json.loads((R / "cost_model.json").read_text())
    out.append(f"Cost model: prevalence {cost['defect_prevalence']:.0%}, escape {cost['cost_escape']:.0f}, false reject "
               f"{cost['cost_false_reject']:.0f}, CHECK {cost['cost_review']:.2f}; no inspection = "
               f"{cost['no_inspection_per_1000']:.0f}, 100% manual = {cost['manual_100pct_per_1000']:.0f} per 1,000 parts.")
    return "\n\n".join(out)


def write_block(path: Path, start: str, end: str, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        print(f"markers not found in {path.name}; skipped")
        return
    head, rest = text.split(start, 1)
    _, tail = rest.split(end, 1)
    path.write_text(f"{head}{start}\n{body}\n{end}{tail}", encoding="utf-8")
    print(f"updated {path.name}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true", help="write the tables into README.md and PROJECT_HANDOFF.md")
    args = parser.parse_args()
    body = build()
    if args.write:
        write_block(ROOT / "README.md", "<!-- RESULTS:START -->", "<!-- RESULTS:END -->", "## Results\n\n" + body)
        write_block(ROOT / "PROJECT_HANDOFF.md", "<!-- HANDOFF-RESULTS:START -->", "<!-- HANDOFF-RESULTS:END -->", body)
    else:
        print(body)


if __name__ == "__main__":
    sys.exit(main())
