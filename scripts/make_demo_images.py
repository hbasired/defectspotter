"""Build ``demo_images/``: images to upload during a live demonstration, with the expected outcome of each.

    python scripts/make_demo_images.py --data data/mvtec_ad

Picks from the **operating test split** (never used for any model or threshold choice):
* per product: good parts the deployed model passes, one example of every defect type it rejects (the most typical
  one: median score among correct decisions), and one borderline part that lands in CHECK (if any);
* a lamp-failure version of a good part (camera-health gate demo);
* carpet: the good parts of the current lot used for live re-qualification.
Images are resized to 512 px (plenty for the 224/288 px models, small enough to upload quickly).
MVTec AD is CC BY-NC-SA 4.0 - research and teaching only.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from defectspotter.data import read_rgb, resize_rgb  # noqa: E402
from defectspotter.drift import apply_drift  # noqa: E402


def decision(score: float, threshold: float, band: float) -> str:
    return "PASS" if score < threshold - band else ("CHECK" if score < threshold + band else "REJECT")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", default="data/mvtec_ad")
    parser.add_argument("--out", default=str(ROOT / "demo_images"))
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    data, out = Path(args.data), Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)

    scores = json.loads((ROOT / "reports" / "scores.json").read_text())
    requal = json.loads((ROOT / "reports" / "requalification_scores.json").read_text()) \
        if (ROOT / "reports" / "requalification_scores.json").exists() else {}
    req_table = pd.read_csv(ROOT / "reports" / "requalification.csv") if (ROOT / "reports" / "requalification.csv").exists() else pd.DataFrame()
    splits = pd.read_csv(ROOT / "reports" / "splits.csv", keep_default_na=False)
    rows = []

    def save(image: np.ndarray, folder: str, name: str) -> str:
        target = out / folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(resize_rgb(image, args.size)).save(target)
        return f"{folder}/{name}"

    for category, info in scores.items():
        test = splits[(splits.category == category) & (splits.split == "test")].reset_index(drop=True)
        roles = np.asarray(info["test_roles"])
        op = roles == "operating_test"
        cal = json.loads((ROOT / "artifacts" / category / "calibration.json").read_text())
        deployed = (not req_table.empty and category in set(req_table.category)
                    and bool(req_table[req_table.category == category].iloc[0].deployed))
        if deployed:  # current lot re-qualified: use the re-qualified scores and policy
            d = requal[category]
            op_scores = np.asarray(d["operating_scores_requalified"])
            threshold, band, model = d["threshold"], d["band"], "current lot (re-qualified)"
        else:
            op_scores = np.asarray(info["methods"]["patchcore"]["test"])[op]
            pol = cal["methods"]["patchcore"]["policies"]["synthetic_cost"]
            threshold, band, model = pol["threshold"], pol["band"], "deployed PatchCore"
        frame = test[op].reset_index(drop=True).assign(score=op_scores)
        frame["decision"] = [decision(s, threshold, band) for s in frame.score]
        frame["correct"] = np.where(frame.label == 0, frame.decision == "PASS", frame.decision == "REJECT")
        picks = []
        goods = frame[(frame.label == 0) & frame.correct].sort_values("score")
        picks += [goods.iloc[i] for i in sorted({0, len(goods) // 2}) if len(goods) > i]
        for defect_type, group in frame[frame.label == 1].groupby("defect_type"):
            ok = group[group.correct].sort_values("score")
            if len(ok):
                picks.append(ok.iloc[len(ok) // 2])
        border = frame[frame.decision == "CHECK"]
        if len(border):
            picks.append(border.iloc[0])
        for row in picks:
            kind = "good" if row.label == 0 else f"defect_{row.defect_type}"
            tag = "_borderline" if row.decision == "CHECK" else ""
            name = f"{kind}{tag}_{Path(row.image_path).stem}.png"
            path = save(read_rgb(data / row.image_path), category, name)
            rows.append({"file": path, "product": category, "ground truth": "good" if row.label == 0 else
                         f"defective: {row.defect_type.replace('_', ' ')}", "expected decision": row.decision,
                         "expected score": round(float(row.score), 1), "model": model,
                         "what to show": "passes cleanly" if row.label == 0 else (
                             "borderline → human review" if row.decision == "CHECK" else "rejected; heat map on the defect")})
        # camera-health demo: a good part under a failing lamp
        if len(goods):
            dim = apply_drift(read_rgb(data / goods.iloc[0].image_path), "lighting_dim", 1.0)
            path = save(dim, "drift_demo", f"{category}_good_part_lamp_failure.png")
            rows.append({"file": path, "product": category, "ground truth": "good (lamp failure simulated)",
                         "expected decision": "CHECK", "expected score": "high", "model": model,
                         "what to show": "camera-health gate: no automatic REJECT on an unqualified image"})

    # carpet / any re-qualified product: the current-lot good parts used for live re-qualification
    for category in (req_table[req_table.deployed].category if not req_table.empty else []):
        test = splits[(splits.category == category) & (splits.split == "test")]
        for k, p in enumerate(test[(test.role == "oracle_calibration") & (test.label == 0)].image_path):
            save(read_rgb(data / p), f"{category}_new_lot_good_parts", f"current_lot_good_{k:02d}.png")
        rows.append({"file": f"{category}_new_lot_good_parts/ (all files)", "product": category,
                     "ground truth": "good parts of the current lot", "expected decision": "—", "expected score": "—",
                     "model": "re-qualification input",
                     "what to show": "Drift & new lots → section C: upload them all → Re-qualify"})

    table = pd.DataFrame(rows)
    table.to_csv(out / "expected_outcomes.csv", index=False)
    lines = ["# Demo images", "",
             "Upload these during a live demonstration (🎬 Live demo → Inspect a part, after selecting the product).",
             "They come from the MVTec AD **operating test split** — never used to train a model or choose a threshold —",
             "so the expected outcomes below are honest. Images: CC BY-NC-SA 4.0 (MVTec AD), research and teaching only.", "",
             "| file | product | ground truth | expected decision | expected score | what to show |", "|---|---|---|---|---|---|"]
    lines += [f"| `{r['file']}` | {r['product']} | {r['ground truth']} | **{r['expected decision']}** | {r['expected score']} | "
              f"{r['what to show']} |" for r in rows]
    lines += ["", "Scores are on the 0–100 scale (25 ≈ typical good part, 75 ≈ typical synthetic defect). Decisions use the "
              "deployed thresholds; carpet uses the model re-qualified for the current lot. Small differences can occur "
              "with resized uploads."]
    (out / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with zipfile.ZipFile(ROOT / "demo_images.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(out.parent))
    print(table.to_string(index=False))
    print(f"\n{len(table)} entries -> {out} (+ demo_images.zip)")


if __name__ == "__main__":
    main()
