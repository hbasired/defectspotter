"""Everything the guide quotes, read from reports/ and artifacts/ (so the guide never disagrees with the models)."""
from __future__ import annotations

import base64
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

LABELS = {"patchcore": "PatchCore", "fvae": "Feature-space VAE", "vae": "Pixel VAE", "mean": "Mean-image baseline",
          "ensemble": "Ensemble"}
POLICY_LABELS = {"synthetic_cost": "synthetic defects (deployed default)",
                 "synthetic_plus_diffusion_cost": "synthetic + diffusion defects",
                 "diffusion_cost": "diffusion defects only",
                 "synthetic_conspicuous_cost": "full-strength synthetic only (no smoothing)",
                 "good_quantile": "1% quantile of good parts (no defects at all)",
                 "real_oracle_cost": "real labelled defects (oracle, comparison only)"}


class GuideData:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        r = self.root / "reports"
        rd = lambda name: pd.read_csv(r / name)  # noqa: E731
        self.summary = json.loads((r / "summary.json").read_text())
        self.cats: list[str] = list(self.summary["categories"])
        self.metrics = rd("metrics.csv")
        self.ops = rd("operating_metrics.csv")
        self.requal = rd("requalification.csv")
        self.req_scores = json.loads((r / "requalification_scores.json").read_text())
        self.latency = rd("latency.csv")
        self.few_shot = rd("few_shot.csv")
        self.drift_sens = rd("drift_sensitivity.csv")
        self.drift_alarms = rd("drift_alarms.csv")
        self.sizes = rd("artifact_sizes.csv")
        self.composition = rd("dataset_composition.csv")
        self.splits = rd("splits.csv")
        self.synthetic = rd("synthetic_summary.csv")
        self.diffusion = rd("diffusion_summary.csv")
        self.thresholds = rd("threshold_comparison.csv")
        self.backbone = rd("backbone_study.csv")
        self.capability = rd("detection_capability.csv")
        self.hpo = rd("hpo_patchcore.csv")
        self.per_type = rd("per_defect_type.csv")
        self.fvae_hist = rd("fvae_training_history.csv")
        self.vae_hist = rd("vae_training_history.csv")
        self.cost = json.loads((r / "cost_model.json").read_text())
        self.manifest = json.loads((self.root / "artifacts" / "manifest.json").read_text())
        self.cal = {c: json.loads((self.root / "artifacts" / c / "calibration.json").read_text()) for c in self.cats}
        self.mean_cfg = {c: json.loads((self.root / "artifacts" / c / "mean_baseline_config.json").read_text())
                         for c in self.cats}
        self.lots = {}
        for c in self.cats:
            p = self.root / "artifacts" / c / "requalified" / "requalification.json"
            if p.exists():
                self.lots[c] = json.loads(p.read_text())
        demo = self.root / "demo_images" / "expected_outcomes.csv"
        self.demo = pd.read_csv(demo) if demo.exists() else pd.DataFrame()

    # ----------------------------------------------------------------------------------------------- lookups
    def metric(self, cat: str, method: str, col: str = "image_auroc") -> float:
        row = self.metrics[(self.metrics.category == cat) & (self.metrics.method == method)]
        return float(row.iloc[0][col]) if len(row) else float("nan")

    def macro(self, method: str, col: str = "image_auroc") -> float:
        return float(self.summary["macro"][method][col])

    def op(self, cat: str, method: str = "patchcore", policy: str = "synthetic_cost") -> pd.Series:
        rows = self.ops[(self.ops.category == cat) & (self.ops.method == method) & (self.ops.policy == policy)]
        return rows.iloc[0]

    def selected_recipe(self, cat: str) -> pd.Series:
        h = self.hpo[self.hpo.category == cat]
        sel = h[h.selected.astype(bool)] if "selected" in h else h.iloc[0:0]
        return sel.iloc[0] if len(sel) else h.sort_values("objective").iloc[-1]

    def policy(self, cat: str, method: str = "patchcore", name: str | None = None) -> dict:
        info = self.cal[cat]["methods"][method]
        return dict(info["policies"][name or info["default_policy"]])

    def deployed(self, cat: str) -> dict:
        """Threshold/band actually used by the app for PatchCore (re-qualified lot where the lot changed)."""
        req = self.requal[self.requal.category == cat]
        if len(req) and bool(req.iloc[0].deployed) and cat in self.lots:
            lot = self.lots[cat]
            return {"threshold": float(lot["threshold"]), "band": float(lot["band"]), "lot": True}
        p = self.policy(cat)
        return {"threshold": float(p["threshold"]), "band": float(p["band"]), "lot": False}

    def roles(self) -> pd.DataFrame:
        return self.splits.groupby(["category", "role"]).size().unstack(fill_value=0)

    def bank_vectors(self, cat: str) -> int:
        return int(self.selected_recipe(cat).bank_vectors)

    # ----------------------------------------------------------------------------------------- derived numbers
    def numbers(self) -> dict:
        roles = self.roles()
        totals = roles.sum()
        cfg = self.summary["patchcore_configs"]
        grid = {c: (int(cfg[c]["image_size"]) // 8) ** 2 * int(cfg[c].get("bank_rotations", 1)) for c in self.cats}
        train = {c: int(roles.loc[c, "model_train_good"]) for c in self.cats}
        banks = [self.bank_vectors(c) for c in self.cats]
        return {
            "artifact_mb": float(self.summary["artifact_total_mb"]),
            "bank_range": f"{min(banks):,}–{max(banks):,}",
            "bank_counts": " / ".join(f"{b:,}" for b in banks),
            "train_counts": " / ".join(f"{train[c]:,}" for c in self.cats) + " good images",
            "patch_counts": " / ".join(f"{train[c] * grid[c]:,}" for c in self.cats),
            "calib_counts": " / ".join(str(int(roles.loc[c, "calibration_good"])) for c in self.cats),
            "roles": {k: int(v) for k, v in totals.items()},
            "train_total": int(totals.get("model_train_good", 0) + totals.get("calibration_good", 0)),
            "test_total": int(totals.get("oracle_calibration", 0) + totals.get("operating_test", 0)),
        }

    def real_world(self):
        """(rows, scenario) exactly as the app's Overview computes them."""
        import sys
        sys.path.insert(0, str(self.root))
        from defectspotter.operations import LineScenario
        from ui.explain import real_world_rows
        from ui.store import ReportStore
        rows = real_world_rows(ReportStore(self.root), SimpleNamespace(categories=self.cats))
        return rows, LineScenario()


def raw_uri(path: Path) -> str:
    """Embed an already-compressed WebP/JPEG/PNG file unchanged."""
    mime = {".webp": "image/webp", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png"}[path.suffix.lower()]
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def nan_to_dash(x, fmt="{:.3f}") -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else fmt.format(x)
