"""Read-only access to the evidence produced by the training pipeline (reports/, examples/)."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


class ReportStore:
    def __init__(self, project_root: str | Path) -> None:
        self.root = Path(project_root)
        self.reports = self.root / "reports"
        self.figures = self.reports / "figures"
        self.examples = self.root / "examples"

    # ------------------------------------------------------------- loaders
    @lru_cache(maxsize=64)
    def csv(self, name: str) -> pd.DataFrame:
        path = self.reports / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path, keep_default_na=True)

    @lru_cache(maxsize=32)
    def json(self, name: str) -> dict:
        path = self.reports / name
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    @property
    def summary(self) -> dict:
        return self.json("summary.json")

    def figure(self, name: str) -> str | None:
        path = self.figures / name
        return str(path) if path.exists() else None

    # ------------------------------------------------------------ examples
    @lru_cache(maxsize=1)
    def example_manifest(self) -> pd.DataFrame:
        path = self.examples / "manifest.csv"
        if not path.exists():
            return pd.DataFrame(columns=["category", "file", "mask", "label", "defect_type"])
        return pd.read_csv(path, keep_default_na=False)

    def examples_for(self, category: str) -> pd.DataFrame:
        frame = self.example_manifest()
        frame = frame[frame.category == category].copy()
        # good parts first, then defects alphabetically - stable order for the gallery
        frame["order"] = np.where(frame.label == 0, 0, 1)
        return frame.sort_values(["order", "defect_type", "file"]).reset_index(drop=True)

    def example_gallery(self, category: str) -> list[tuple[str, str]]:
        return [(str(self.examples / row.file), "good" if row.label == 0 else row.defect_type.replace("_", " "))
                for row in self.examples_for(category).itertuples()]

    def synthetic_gallery(self) -> list[tuple[str, str]]:
        folder = self.examples / "synthetic"
        if not folder.exists():
            return []
        items = []
        for path in sorted(folder.glob("*.jpg")):
            parts = path.stem.split("_")
            items.append((str(path), f"{parts[0]} · {parts[1]}"))
        return items

    def diffusion_gallery(self) -> list[tuple[str, str]]:
        folder = self.examples / "diffusion"
        if not folder.exists():
            return []
        return [(str(p), p.stem.replace("_", " ")) for p in sorted(folder.glob("*.jpg"))]

    def new_product_samples(self, kind: str) -> list[str]:
        folder = self.examples / "new_product"
        if not folder.exists():
            return []
        return [str(p) for p in sorted(folder.glob(f"*/{kind}/*.png"))]

    # -------------------------------------------------------------- scores
    def score_groups(self, category: str, method: str) -> dict:
        """Normalised scores of every evaluation group for the threshold lab."""
        data = self.json("scores.json").get(category)
        if not data or method not in data.get("methods", {}):
            return {}
        m = data["methods"][method]
        labels = np.asarray(data["test_labels"])
        roles = np.asarray(data["test_roles"])
        test = np.asarray(m["test"])
        groups = {
            "cal_good": np.asarray(m.get("cal", [])),
            "synthetic": np.asarray(m.get("syn", [])),
            "test_scores": test, "test_labels": labels, "test_roles": roles,
            "test_types": np.asarray(data["test_defect_types"]),
        }
        if "diff" in m:
            groups["diffusion"] = np.asarray(m["diff"])
        if "sev" in m and data.get("sev_levels"):
            groups["severity"] = np.asarray(m["sev"])
            groups["severity_levels"] = np.asarray(data["sev_levels"])
        return groups

    @staticmethod
    def calibration_groups(groups: dict) -> dict:
        """Map threshold-lab groups to the keys of ``decision.calibration_sets`` (label-free data only)."""
        return {"cal": groups["cal_good"], "syn": groups["synthetic"], "diff": groups.get("diffusion"),
                "sev": groups.get("severity"), "sev_levels": groups.get("severity_levels")}
