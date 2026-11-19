"""Print the headline result tables (Markdown) from reports/ - used to keep README numbers exact."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "reports"
LABELS = {"patchcore": "PatchCore", "vae": "VAE", "mean": "Mean baseline", "ensemble": "Ensemble"}


def table(frame: pd.DataFrame) -> str:
    cols = list(frame.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(v) for v in row.values) + " |")
    return "\n".join(lines)


def main() -> None:
    metrics = pd.read_csv(R / "metrics.csv")
    out = []
    m = metrics.copy()
    m["image AUROC [95% CI]"] = m.apply(lambda r: f"{r.image_auroc:.3f} [{r.image_auroc_ci_low:.3f}–{r.image_auroc_ci_high:.3f}]", axis=1)
    m["pixel AUROC"] = m.pixel_auroc.map("{:.3f}".format)
    m["AUPRO@30%"] = m.aupro_30.map("{:.3f}".format)
    m["method"] = m.method.map(LABELS)
    out.append("### Detection and localisation (MVTec test images)\n" + table(m[["category", "method", "image AUROC [95% CI]", "pixel AUROC", "AUPRO@30%"]]))
    macro = metrics.groupby("method")[["image_auroc", "pixel_auroc", "aupro_30"]].mean().round(3).reset_index()
    macro["method"] = macro.method.map(LABELS)
    out.append("### Mean over the three products\n" + table(macro))
    ops = pd.read_csv(R / "operating_metrics.csv")
    o = ops[ops.policy.isin(["synthetic_cost", "real_oracle_cost"])].copy()
    o = o[o.method.isin(["patchcore", "vae", "ensemble"])]
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
    t = thr[thr.method.isin(["patchcore", "vae"])][cols].round(1)
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
    print("\n\n".join(out))


if __name__ == "__main__":
    sys.exit(main())
