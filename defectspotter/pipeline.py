"""End-to-end experiment: EDA -> synthetic defects -> train -> calibrate -> evaluate -> export.

Run with ``python scripts/train.py --data /path/to/mvtec`` (see ``--help``).
Every stage writes plain CSV/JSON/PNG evidence to ``reports/`` and deployable
models to ``artifacts/``; the app only reads those folders. Real defect images
are never used to fit a detector or to choose the default threshold - they are
used only for evaluation and for the "oracle" threshold comparison.
"""
from __future__ import annotations

import json
import platform
import shutil
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from . import __version__
from .baseline import MeanBaselineConfig, MeanImageBaseline
from .constants import CATEGORY_INFO, FEW_SHOT_CATEGORY, MAIN_CATEGORIES, MVTEC_HF_REPO, MVTEC_HF_REVISION
from .data import (acquisition_stats, index_mvtec, load_json, make_splits, read_mask, read_rgb, resize_rgb,
                   save_json)
from .decision import (CostModel, fit_pixel_scale, fit_score_scale, normalize_map, normalize_scores,
                       operating_metrics, reference_costs, select_threshold_by_cost,
                       select_threshold_good_quantile, decide_many)
from .drift import DRIFT_KINDS, EWMAChart, apply_drift, first_alarm, health_check, reference_from_stats
from .metrics import aupro, bootstrap_auroc_ci, pixel_auroc, roc_points, safe_ap, safe_auroc
from .patchcore import MemoryBank, PatchCoreConfig, build_memory_bank, score_grid, score_grids
from .synthetic import generate_synthetic_set
from .vae import VAEConfig, score_reconstruction

METHODS = ("patchcore", "vae", "mean", "ensemble")
MAP = 224  # common resolution for all evaluation maps


@dataclass
class PipelineConfig:
    data_root: str
    project_root: str = "."
    work_dir: str = "../defectspotter_work"
    categories: tuple = MAIN_CATEGORIES
    few_shot_category: str | None = FEW_SHOT_CATEGORY
    backbone: str = "resnet18"
    image_size: int = 224
    coreset_grid: tuple = (0.01, 0.05, 0.10)
    blur_grid: tuple = (2.0, 4.0)
    score_modes: tuple = ("max", "topk")
    topk_grid: tuple = (0.01, 0.05)
    max_bank_mb: float = 45.0
    vae_image_size: int = 128
    vae_latent_dim: int = 128
    vae_base_channels: int = 32
    vae_epochs: int = 60
    synthetic_per_image: int = 2
    bootstrap: int = 1000
    seed: int = 42
    device: str = "cpu"
    costs: CostModel = field(default_factory=CostModel)
    stages: tuple = ("index", "eda", "synthetic", "patchcore", "vae", "mean", "severity", "diffusion", "calibrate", "evaluate",
                     "latency", "drift", "few_shot", "examples", "summary")

    @property
    def artifacts(self) -> Path:
        return Path(self.project_root) / "artifacts"

    @property
    def reports(self) -> Path:
        return Path(self.project_root) / "reports"

    @property
    def figures(self) -> Path:
        return Path(self.project_root) / "reports" / "figures"

    @property
    def examples(self) -> Path:
        return Path(self.project_root) / "examples"

    @property
    def work(self) -> Path:
        return Path(self.work_dir)


def _alarm_count(chart: pd.DataFrame, start: int, end: int) -> int:
    window = chart.iloc[start:end]
    return int((window.alarm_high | window.alarm_low).sum())


def _log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


class Pipeline:
    def __init__(self, config: PipelineConfig, progress: Callable[[str], None] | None = None) -> None:
        self.cfg = config
        self.log = progress or _log
        for folder in (config.artifacts, config.reports, config.figures, config.examples, config.work):
            folder.mkdir(parents=True, exist_ok=True)
        self.data_root = Path(config.data_root).resolve()
        self._extractor = None

    # ------------------------------------------------------------ helpers
    def rel(self, path: str) -> str:
        try:
            return str(Path(path).resolve().relative_to(self.data_root))
        except ValueError:
            return Path(path).name

    def abs(self, rel_path: str) -> str:
        return str(self.data_root / rel_path)

    @property
    def extractor(self):
        if self._extractor is None:
            from .features import PatchFeatureExtractor

            self._extractor = PatchFeatureExtractor(self.cfg.backbone).to(self.cfg.device).eval()
        return self._extractor

    def splits(self) -> pd.DataFrame:
        return pd.read_csv(self.cfg.reports / "splits.csv", keep_default_na=False)

    def paths(self, category: str, role: str | None = None, split: str | None = None) -> pd.DataFrame:
        frame = self.splits()
        frame = frame[frame.category == category]
        if role:
            frame = frame[frame.role == role]
        if split:
            frame = frame[frame.split == split]
        return frame.reset_index(drop=True)

    def synthetic(self, category: str) -> pd.DataFrame:
        return pd.read_csv(self.cfg.work / "synthetic" / category / "manifest.csv")

    def diffusion(self, category: str) -> pd.DataFrame | None:
        manifest = self.cfg.work / "diffusion" / category / "manifest.csv"
        return pd.read_csv(manifest) if manifest.exists() else None

    def test_masks(self, category: str) -> list[np.ndarray]:
        test = self.paths(category, split="test")
        return [read_mask(self.abs(p) if p else None, MAP) for p in test.mask_path]

    # --------------------------------------------------------------- stages
    def run(self) -> None:
        started = time.time()
        for stage in self.cfg.stages:
            if stage == "few_shot" and not self.cfg.few_shot_category:
                continue
            t0 = time.time()
            self.log(f"=== stage: {stage}")
            getattr(self, f"stage_{stage}")()
            self.log(f"=== stage {stage} done in {time.time() - t0:.0f}s")
        self.log(f"Pipeline finished in {(time.time() - started) / 60:.1f} min")

    def stage_index(self) -> None:
        categories = list(self.cfg.categories) + ([self.cfg.few_shot_category] if self.cfg.few_shot_category else [])
        index = index_mvtec(self.data_root, categories)
        splits = make_splits(index, seed=self.cfg.seed)
        splits["image_path"] = splits.image_path.map(self.rel)
        splits["mask_path"] = splits.mask_path.map(lambda p: self.rel(p) if p else "")
        splits.to_csv(self.cfg.reports / "splits.csv", index=False)
        counts = splits.groupby(["category", "role", "label"]).size().rename("images").reset_index()
        self.log("\n" + counts.to_string(index=False))

    def stage_eda(self) -> None:
        splits = self.splits()
        rows = []
        location = {}
        for record in splits.to_dict("records"):
            image = read_rgb(self.abs(record["image_path"]))
            stats = acquisition_stats(image)
            mask_area, regions = 0.0, 0
            if record["mask_path"]:
                mask = read_mask(self.abs(record["mask_path"]))
                mask_area = float(100 * mask.mean())
                regions = int(cv2.connectedComponents(mask, connectivity=8)[0] - 1)
                acc = location.setdefault(record["category"], np.zeros((128, 128), np.float32))
                acc += cv2.resize(mask.astype(np.float32), (128, 128), interpolation=cv2.INTER_AREA)
            rows.append({**{k: record[k] for k in ("category", "split", "role", "label", "defect_type", "image_path")},
                         "width": image.shape[1], "height": image.shape[0], **stats,
                         "mask_area_pct": mask_area, "defect_regions": regions})
        stats = pd.DataFrame(rows)
        stats.to_csv(self.cfg.reports / "image_stats.csv", index=False)
        composition = (stats.groupby(["category", "split", "defect_type"]).size().rename("images").reset_index())
        composition.to_csv(self.cfg.reports / "dataset_composition.csv", index=False)
        for category, acc in location.items():
            heat = acc / max(acc.max(), 1e-6)
            colored = cv2.applyColorMap((heat * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)
            cv2.imwrite(str(self.cfg.figures / f"eda_defect_location_{category}.png"),
                        cv2.resize(colored, (256, 256), interpolation=cv2.INTER_NEAREST))
        for category in splits.category.unique():
            self._montage(category)
        self.log(f"EDA: {len(stats)} images profiled")

    def _montage(self, category: str, tile: int = 220) -> None:
        """Good samples + one example per defect type with the ground-truth outline."""
        frame = self.paths(category)
        tiles, captions = [], []
        good = frame[(frame.split == "train")].sample(n=3, random_state=self.cfg.seed)
        for record in good.itertuples():
            tiles.append(resize_rgb(read_rgb(self.abs(record.image_path)), tile))
            captions.append("good")
        for defect_type, group in frame[frame.label == 1].groupby("defect_type"):
            record = group.iloc[0]
            image = resize_rgb(read_rgb(self.abs(record.image_path)), tile)
            mask = read_mask(self.abs(record.mask_path), tile) if record.mask_path else np.zeros((tile, tile), np.uint8)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            image = np.ascontiguousarray(image.copy())
            cv2.drawContours(image, contours, -1, (255, 40, 40), 2)
            tiles.append(image)
            captions.append(defect_type.replace("_", " "))
        cols = 4
        rows = int(np.ceil(len(tiles) / cols))
        canvas = np.full((rows * (tile + 26), cols * tile, 3), 255, np.uint8)
        for i, (image, caption) in enumerate(zip(tiles, captions)):
            r, c = divmod(i, cols)
            canvas[r * (tile + 26) + 26:(r + 1) * (tile + 26), c * tile:(c + 1) * tile] = image
            cv2.putText(canvas, caption, (c * tile + 6, r * (tile + 26) + 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (30, 41, 59), 1, cv2.LINE_AA)
        Image.fromarray(canvas).save(self.cfg.figures / f"eda_montage_{category}.jpg", quality=90)

    def stage_synthetic(self) -> None:
        rows = []
        for category in self.cfg.categories:
            good = self.paths(category, role="calibration_good")
            out = self.cfg.work / "synthetic" / category
            if out.exists():
                shutil.rmtree(out)
            manifest = generate_synthetic_set([self.abs(p) for p in good.image_path], category, out,
                                              per_image=self.cfg.synthetic_per_image, seed=self.cfg.seed)
            rows.append(manifest.assign(image_path=manifest.image_path.map(lambda p: Path(p).name)))
            gallery = self.cfg.examples / "synthetic"
            gallery.mkdir(parents=True, exist_ok=True)
            for kind, group in manifest.groupby("synthetic_type"):
                for k, record in enumerate(group.head(2).itertuples()):
                    image = resize_rgb(read_rgb(record.image_path), 320)
                    mask = read_mask(record.mask_path, 320)
                    Image.fromarray(image).save(gallery / f"{category}_{kind}_{k}.jpg", quality=92)
                    Image.fromarray(mask * 255).save(gallery / f"{category}_{kind}_{k}_mask.png")
        frame = pd.concat(rows, ignore_index=True)
        frame[["category", "synthetic_type", "mask_area_pct", "seed"]].to_csv(
            self.cfg.reports / "synthetic_summary.csv", index=False)
        self.log(f"Synthetic defects: {len(frame)} images")

    # ----------------------------------------------------------- PatchCore
    def _features(self, category: str) -> dict:
        from .features import extract_features

        cache = self.cfg.work / "features" / f"{category}_{self.cfg.backbone}_{self.cfg.image_size}.npz"
        if cache.exists():
            return dict(np.load(cache))
        sets = {
            "train": [self.abs(p) for p in self.paths(category, role="model_train_good").image_path],
            "cal": [self.abs(p) for p in self.paths(category, role="calibration_good").image_path],
            "test": [self.abs(p) for p in self.paths(category, split="test").image_path],
        }
        if category in self.cfg.categories:
            sets["syn"] = self.synthetic(category).image_path.tolist()
        out = {}
        for name, items in sets.items():
            self.log(f"  features {category}/{name}: {len(items)} images")
            out[name] = extract_features(self.extractor, items, self.cfg.image_size, device=self.cfg.device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **out)
        return out

    def stage_patchcore(self) -> None:
        hpo_rows = []
        for category in self.cfg.categories:
            feats = self._features(category)
            syn = self.synthetic(category)
            syn_masks = [read_mask(p, MAP) for p in syn.mask_path]
            best = None
            scoring = [("max", 0.01)] if "max" in self.cfg.score_modes else []
            if "topk" in self.cfg.score_modes:
                scoring += [("topk", f) for f in self.cfg.topk_grid]
            for ratio in self.cfg.coreset_grid:
                bank = MemoryBank(build_memory_bank(feats["train"], ratio, self.cfg.max_bank_mb, seed=self.cfg.seed))
                for blur in self.cfg.blur_grid:
                    for mode, topk in scoring:
                        config = PatchCoreConfig(backbone=self.cfg.backbone, image_size=self.cfg.image_size,
                                                 coreset_ratio=ratio, blur_sigma=blur, score_mode=mode,
                                                 topk_fraction=topk)
                        good_s, good_m = score_grids(feats["cal"], bank, config)
                        syn_s, syn_m = score_grids(feats["syn"], bank, config)
                        labels = np.r_[np.zeros(len(good_s)), np.ones(len(syn_s))]
                        image_auc = safe_auroc(labels, np.r_[good_s, syn_s])
                        # AUROC saturates on synthetic data; the share of synthetic defects caught above
                        # the 99th percentile of good scores (TPR at <=1% false rejects) does not, and it is
                        # the operating requirement of a real line.
                        tpr_at_1pct = float(np.mean(syn_s > np.quantile(good_s, 0.99)))
                        pix_auc = pixel_auroc([*good_m, *syn_m],
                                              [np.zeros((MAP, MAP), np.uint8)] * len(good_m) + syn_masks)
                        bank_mb = len(bank) * bank.bank.shape[1] * 2 / 1024 ** 2
                        objective = 0.4 * image_auc + 0.3 * tpr_at_1pct + 0.3 * pix_auc - 1e-4 * np.log1p(bank_mb)
                        row = {"category": category, "coreset_ratio": ratio, "blur_sigma": blur, "score_mode": mode,
                               "topk_fraction": topk if mode == "topk" else None,
                               "bank_vectors": len(bank), "bank_mb": len(bank) * bank.bank.shape[1] * 2 / 1024 ** 2,
                               "synthetic_image_auroc": image_auc, "synthetic_tpr_at_1pct_fpr": tpr_at_1pct,
                               "synthetic_pixel_auroc": pix_auc, "objective": objective}
                        hpo_rows.append(row)
                        if best is None or objective > best[0]:
                            best = (objective, config)
            config = best[1]
            self.log(f"  {category}: selected {config}")
            bank_vectors = build_memory_bank(feats["train"], config.coreset_ratio, self.cfg.max_bank_mb,
                                             seed=self.cfg.seed)
            folder = self.cfg.artifacts / category
            folder.mkdir(parents=True, exist_ok=True)
            np.save(folder / "patchcore_bank.npy", bank_vectors)
            bank = MemoryBank(bank_vectors)
            payload = {"config": config.to_dict()}
            for name in [k for k in ("cal", "syn", "test") if k in feats]:
                scores, maps = score_grids(feats[name], bank, config)
                payload[f"{name}_scores"], payload[f"{name}_maps"] = scores, maps.astype(np.float16)
            train_scores, _ = score_grids(feats["train"][:24], bank, config)  # in-sample sanity only
            payload["train_insample_scores"] = train_scores
            self._save_scores("patchcore", category, payload)
        pd.DataFrame(hpo_rows).to_csv(self.cfg.reports / "hpo_patchcore.csv", index=False)

    def _save_scores(self, method: str, category: str, payload: dict) -> None:
        folder = self.cfg.work / "scores"
        folder.mkdir(parents=True, exist_ok=True)
        config = payload.pop("config", None)
        np.savez_compressed(folder / f"{method}_{category}.npz", **payload)
        if config is not None:
            save_json(config, folder / f"{method}_{category}_config.json")

    def _load_scores(self, method: str, category: str) -> tuple[dict, dict]:
        folder = self.cfg.work / "scores"
        data = dict(np.load(folder / f"{method}_{category}.npz"))
        config = load_json(folder / f"{method}_{category}_config.json", {})
        return data, config

    # ----------------------------------------------------------------- VAE
    def stage_vae(self) -> None:
        import onnxruntime as ort

        from .vae import export_vae_onnx, load_vae_input, train_vae

        rows = []
        for category in self.cfg.categories:
            train = [self.abs(p) for p in self.paths(category, role="model_train_good").image_path]
            rng = np.random.default_rng(self.cfg.seed)
            order = rng.permutation(len(train))
            n_val = max(4, len(train) // 10)
            val_items = [train[i] for i in order[:n_val]]
            fit_items = [train[i] for i in order[n_val:]]
            texture = CATEGORY_INFO.get(category, {}).get("kind") == "texture"
            config = VAEConfig(image_size=self.cfg.vae_image_size, latent_dim=self.cfg.vae_latent_dim,
                               base_channels=self.cfg.vae_base_channels, epochs=self.cfg.vae_epochs,
                               hflip=category != "screw",  # flipping a screw changes thread handedness
                               rot90=True if texture or category in {"bottle", "screw", "metal_nut"} else False)
            self.log(f"  training VAE for {category} on {len(fit_items)} good images")
            model = train_vae(fit_items, val_items, config, seed=self.cfg.seed, device=self.cfg.device,
                              progress=lambda r: self.log(f"    epoch {r['epoch']:3d} train={r['train_loss']:.4f} "
                                                          f"val={r['val_loss']:.4f} ({r['seconds']:.1f}s)")
                              if r["epoch"] % 10 == 0 or r["epoch"] == 1 else None)
            for record in config.history:
                rows.append({"category": category, **record})
            folder = self.cfg.artifacts / category
            report = export_vae_onnx(model, config, folder / "vae.onnx")
            self.log(f"  exported {report}")
            session = ort.InferenceSession(str(folder / "vae.onnx"), providers=["CPUExecutionProvider"])
            sets = {"cal": [self.abs(p) for p in self.paths(category, role="calibration_good").image_path],
                    "syn": self.synthetic(category).image_path.tolist(),
                    "test": [self.abs(p) for p in self.paths(category, split="test").image_path]}
            payload = {"config": config.to_dict()}
            payload_onnx = report
            for name, items in sets.items():
                scores, maps = [], []
                for item in items:
                    x = load_vae_input(item, config.image_size)
                    recon = session.run(None, {"image": x[None]})[0][0]
                    score, amap = score_reconstruction(x, recon, config, output_size=MAP)
                    scores.append(score)
                    maps.append(amap)
                payload[f"{name}_scores"] = np.asarray(scores, np.float32)
                payload[f"{name}_maps"] = np.stack(maps).astype(np.float16)
            self._save_scores("vae", category, payload)
            save_json(payload_onnx, self.cfg.work / "scores" / f"vae_{category}_onnx.json")
        pd.DataFrame(rows).to_csv(self.cfg.reports / "vae_training_history.csv", index=False)

    # ---------------------------------------------------------------- mean
    def stage_mean(self) -> None:
        hpo_rows = []
        for category in self.cfg.categories:
            train = [self.abs(p) for p in self.paths(category, role="model_train_good").image_path]
            cal = [self.abs(p) for p in self.paths(category, role="calibration_good").image_path]
            syn = self.synthetic(category)
            texture = CATEGORY_INFO.get(category, {}).get("kind") == "texture"
            modes = ["none", "translation"] if texture else ["none", "translation", "euclidean", "affine"]
            best = None
            for mode in modes:
                for floor in (4.0, 8.0):
                    model = MeanImageBaseline(MeanBaselineConfig(image_size=MAP, alignment=mode, std_floor=floor))
                    model.fit(train)
                    good = [model.predict(p)[0] for p in cal]
                    bad = [model.predict(p)[0] for p in syn.image_path]
                    auc = safe_auroc(np.r_[np.zeros(len(good)), np.ones(len(bad))], np.r_[good, bad])
                    corr = np.asarray(model.correlations, np.float32)
                    hpo_rows.append({"category": category, "alignment": mode, "std_floor": floor,
                                     "synthetic_image_auroc": auc,
                                     "median_alignment_correlation": float(np.nanmedian(corr)) if np.isfinite(corr).any() else float("nan")})
                    if best is None or auc > best[0]:
                        best = (auc, mode, floor)
            self.log(f"  {category}: mean baseline alignment={best[1]} floor={best[2]} (synthetic AUROC {best[0]:.3f})")
            model = MeanImageBaseline(MeanBaselineConfig(image_size=MAP, alignment=best[1], std_floor=best[2])).fit(train)
            model.save(self.cfg.artifacts / category)
            sets = {"cal": cal, "syn": syn.image_path.tolist(),
                    "test": [self.abs(p) for p in self.paths(category, split="test").image_path]}
            payload = {"config": asdict(model.config)}
            for name, items in sets.items():
                results = [model.predict(item) for item in items]
                payload[f"{name}_scores"] = np.asarray([r[0] for r in results], np.float32)
                payload[f"{name}_maps"] = np.stack([r[1] for r in results]).astype(np.float16)
            self._save_scores("mean", category, payload)
            # Visual explanation of the baseline: mean good image and per-pixel variability.
            mean_rgb = np.clip(model.mean, 0, 255).astype(np.uint8)
            std = model.std.mean(axis=2)
            std_rgb = cv2.cvtColor(cv2.applyColorMap(np.clip(std / max(std.max(), 1e-6) * 255, 0, 255).astype(np.uint8),
                                                     cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            Image.fromarray(np.concatenate([mean_rgb, std_rgb], axis=1)).save(
                self.cfg.figures / f"baseline_mean_std_{category}.jpg", quality=92)
        pd.DataFrame(hpo_rows).to_csv(self.cfg.reports / "hpo_mean_baseline.csv", index=False)

    # ------------------------------------------- detection-limit calibration
    def stage_severity(self, levels: tuple = (0.25, 0.5)) -> None:
        """Replay every synthetic defect at reduced severity (label-free 'detection-limit' set).

        Inspection specs define the faintest defect that must be caught. Blending each synthetic defect
        with its source good image at 25% / 50% strength spans subtle-to-obvious defects, so the cost-
        optimal threshold is not set by conspicuous synthetic defects only.
        """
        import onnxruntime as ort

        from .features import extract_features
        from .vae import load_vae_input

        for category in self.cfg.categories:
            images, kinds = [], []
            for row in self.synthetic(category).itertuples():
                source = read_rgb(row.source_path).astype(np.float32)
                defect = read_rgb(row.image_path).astype(np.float32)
                for level in levels:
                    images.append(((1 - level) * source + level * defect).clip(0, 255).astype(np.uint8))
                    kinds.append(level)
            data, config = self._load_scores("patchcore", category)
            bank = MemoryBank(np.load(self.cfg.artifacts / category / "patchcore_bank.npy"))
            grids = extract_features(self.extractor, images, self.cfg.image_size, device=self.cfg.device)
            scores, _ = score_grids(grids, bank, PatchCoreConfig(**config))
            self._save_scores("patchcore", category, {**data, "config": config, "sev_scores": scores,
                                                       "sev_levels": np.asarray(kinds, np.float32)})
            data, config = self._load_scores("vae", category)
            vae_config = VAEConfig(**{k: v for k, v in config.items() if k in VAEConfig.__dataclass_fields__})
            session = ort.InferenceSession(str(self.cfg.artifacts / category / "vae.onnx"),
                                           providers=["CPUExecutionProvider"])
            vae_scores = []
            for image in images:
                x = load_vae_input(image, vae_config.image_size)
                vae_scores.append(score_reconstruction(x, session.run(None, {"image": x[None]})[0][0], vae_config)[0])
            self._save_scores("vae", category, {**data, "config": config,
                                                 "sev_scores": np.asarray(vae_scores, np.float32),
                                                 "sev_levels": np.asarray(kinds, np.float32)})
            data, config = self._load_scores("mean", category)
            model = MeanImageBaseline.load(self.cfg.artifacts / category)
            self._save_scores("mean", category, {**data, "config": config,
                                                  "sev_scores": np.asarray([model.predict(i)[0] for i in images], np.float32),
                                                  "sev_levels": np.asarray(kinds, np.float32)})
            self.log(f"  {category}: scored {len(images)} reduced-severity synthetic defects (levels {levels})")

    # ---------------------------------------------------- diffusion defects
    def stage_diffusion(self) -> None:
        """Score diffusion-inpainted synthetic defects (scripts/generate_diffusion_defects.py) if present."""
        import onnxruntime as ort

        from .features import extract_features
        from .vae import load_vae_input

        for category in self.cfg.categories:
            manifest = self.diffusion(category)
            if manifest is None or manifest.empty:
                self.log(f"  {category}: no diffusion-inpainted defects found (optional) - skipping")
                continue
            items = manifest.image_path.tolist()
            # PatchCore
            data, config = self._load_scores("patchcore", category)
            bank = MemoryBank(np.load(self.cfg.artifacts / category / "patchcore_bank.npy"))
            grids = extract_features(self.extractor, items, self.cfg.image_size, device=self.cfg.device)
            scores, maps = score_grids(grids, bank, PatchCoreConfig(**config))
            self._save_scores("patchcore", category, {**data, "config": config, "diff_scores": scores,
                                                       "diff_maps": maps.astype(np.float16)})
            # VAE
            data, config = self._load_scores("vae", category)
            vae_config = VAEConfig(**{k: v for k, v in config.items() if k in VAEConfig.__dataclass_fields__})
            session = ort.InferenceSession(str(self.cfg.artifacts / category / "vae.onnx"),
                                           providers=["CPUExecutionProvider"])
            out = []
            for item in items:
                x = load_vae_input(item, vae_config.image_size)
                out.append(score_reconstruction(x, session.run(None, {"image": x[None]})[0][0], vae_config, MAP))
            self._save_scores("vae", category, {**data, "config": config,
                                                 "diff_scores": np.asarray([o[0] for o in out], np.float32),
                                                 "diff_maps": np.stack([o[1] for o in out]).astype(np.float16)})
            # Mean baseline
            data, config = self._load_scores("mean", category)
            model = MeanImageBaseline.load(self.cfg.artifacts / category)
            out = [model.predict(item) for item in items]
            self._save_scores("mean", category, {**data, "config": config,
                                                  "diff_scores": np.asarray([o[0] for o in out], np.float32),
                                                  "diff_maps": np.stack([o[1] for o in out]).astype(np.float16)})
            self.log(f"  {category}: scored {len(items)} diffusion-inpainted defects")

    # ----------------------------------------------------------- calibrate
    def stage_calibrate(self) -> None:
        stats = pd.read_csv(self.cfg.reports / "image_stats.csv")
        costs = self.cfg.costs
        capability_rows = []
        for category in self.cfg.categories:
            sev_levels = None
            test = self.paths(category, split="test")
            oracle = (test.role == "oracle_calibration").to_numpy()
            labels = test.label.to_numpy()
            methods: dict[str, dict] = {}
            normalized: dict[str, dict] = {}
            syn_masks = [read_mask(p, MAP) for p in self.synthetic(category).mask_path]
            for method in ("patchcore", "vae", "mean"):
                data, config = self._load_scores(method, category)
                scale = fit_score_scale(data["cal_scores"], data["syn_scores"])
                pixel_scale = fit_pixel_scale(data["cal_maps"].astype(np.float32),
                                              synthetic_maps=data["syn_maps"].astype(np.float32),
                                              synthetic_masks=syn_masks)
                normalized[method] = {k[:-7]: normalize_scores(v, scale) for k, v in data.items() if k.endswith("_scores")}
                methods[method] = {"config": config, "score_scale": scale, "pixel_scale": pixel_scale}
                if "sev_levels" in data:
                    sev_levels = data["sev_levels"]
            # Ensemble weight chosen on synthetic calibration data only (label-free).
            best_w, best_auc = 0.5, -1.0
            for w in np.round(np.linspace(0, 1, 11), 2):
                good = w * normalized["patchcore"]["cal"] + (1 - w) * normalized["vae"]["cal"]
                syn = w * normalized["patchcore"]["syn"] + (1 - w) * normalized["vae"]["syn"]
                auc = safe_auroc(np.r_[np.zeros(len(good)), np.ones(len(syn))], np.r_[good, syn])
                if auc > best_auc + 1e-9:
                    best_w, best_auc = float(w), auc
            combined = {k: best_w * normalized["patchcore"][k] + (1 - best_w) * normalized["vae"][k]
                        for k in normalized["patchcore"] if k in normalized["vae"]}
            ens_scale = fit_score_scale(combined["cal"], combined["syn"])
            normalized["ensemble"] = {k: normalize_scores(v, ens_scale) for k, v in combined.items()}
            methods["ensemble"] = {"weight_patchcore": best_w, "synthetic_auroc": best_auc, "score_scale": ens_scale,
                                   "pixel_scale": {"low": 0.0, "threshold": 50.0, "high": 100.0}, "config": {}}
            for method, info in methods.items():
                s = normalized[method]
                syn_labels = np.r_[np.zeros(len(s["cal"])), np.ones(len(s["syn"]))]
                # Label-free capability gate: a calibration set joins the default policy only if the detector
                # separates it from held-out good parts (synthetic AUROC >= 0.9). Weaker sets mark the detection
                # limit and are reported, instead of driving the cost optimum to "reject everything".
                candidate_sets = {"synthetic 100%": s["syn"]}
                if "sev" in s and sev_levels is not None:
                    for level in sorted(set(np.round(sev_levels, 2).tolist()), reverse=True):
                        candidate_sets[f"synthetic {int(round(100 * level))}%"] = s["sev"][np.isclose(sev_levels, level)]
                if "diff" in s:
                    candidate_sets["diffusion-refined"] = s["diff"]
                spec_parts, capability = [], {}
                for name, values in candidate_sets.items():
                    auc = safe_auroc(np.r_[np.zeros(len(s["cal"])), np.ones(len(values))], np.r_[s["cal"], values])
                    used = bool(auc >= 0.9 or name == "synthetic 100%")
                    capability[name] = {"synthetic_auroc": auc, "used_for_default_threshold": used, "n": int(len(values))}
                    capability_rows.append({"category": category, "method": method, "calibration_set": name,
                                            "synthetic_auroc": auc, "n": int(len(values)), "used": used})
                    if used:
                        spec_parts.append(values)
                spec = np.concatenate(spec_parts)
                info["detection_capability"] = capability
                policies = {
                    "synthetic_cost": select_threshold_by_cost(
                        np.r_[s["cal"], spec], np.r_[np.zeros(len(s["cal"])), np.ones(len(spec))], costs),
                    "synthetic_conspicuous_cost": select_threshold_by_cost(np.r_[s["cal"], s["syn"]], syn_labels, costs),
                    "good_quantile": {"threshold": select_threshold_good_quantile(s["cal"], 0.01, 5.0), "band": 5.0},
                    "real_oracle_cost": select_threshold_by_cost(s["test"][oracle], labels[oracle], costs),
                }
                if "diff" in s:
                    diff_labels = np.r_[np.zeros(len(s["cal"])), np.ones(len(s["diff"]))]
                    policies["diffusion_cost"] = select_threshold_by_cost(np.r_[s["cal"], s["diff"]], diff_labels, costs)
                    both = np.r_[s["syn"], s["diff"]]
                    policies["synthetic_plus_diffusion_cost"] = select_threshold_by_cost(
                        np.r_[s["cal"], both], np.r_[np.zeros(len(s["cal"])), np.ones(len(both))], costs)
                info["policies"] = {name: {"threshold": float(p["threshold"]), "band": float(p["band"])}
                                    for name, p in policies.items()}
                info["default_policy"] = "synthetic_cost"
                info["policy_notes"] = {
                    "synthetic_cost": "default: held-out good + every synthetic set the detector can reliably separate "
                                      "(severity 100/50/25% and diffusion-refined; synthetic AUROC >= 0.9)",
                    "synthetic_conspicuous_cost": "held-out good + full-strength synthetic defects only (first version)",
                    "good_quantile": "1% false-reject quantile of held-out good parts (no defects at all)",
                    "real_oracle_cost": "labelled real defects (oracle split) - comparison only, never deployed"}
            reference = reference_from_stats(stats[(stats.category == category) & (stats.role == "model_train_good")])
            calibration = {"category": category, "info": CATEGORY_INFO.get(category, {}), "methods": methods,
                           "health_reference": reference, "costs": costs.to_dict(),
                           "threshold_policy_note": "Default = cost-optimal on held-out good + synthetic defects; "
                                                    "real_oracle_cost uses labelled defects and is for comparison only."}
            save_json(calibration, self.cfg.artifacts / category / "calibration.json")
            save_json({m: {k: v.tolist() for k, v in d.items()} for m, d in normalized.items()},
                      self.cfg.work / "scores" / f"normalized_{category}.json")
            self.log(f"  {category}: ensemble weight={best_w}, policies="
                     f"{ {m: methods[m]['policies']['synthetic_cost'] for m in methods} }")
        pd.DataFrame(capability_rows).to_csv(self.cfg.reports / "detection_capability.csv", index=False)
        self._export_extractor()

    def _export_extractor(self) -> None:
        from .features import count_parameters, export_onnx

        report = export_onnx(self.extractor.cpu(), self.cfg.artifacts / "feature_extractor.onnx", self.cfg.image_size)
        report["parameters"] = count_parameters(self.extractor)
        save_json({"version": __version__, "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "categories": list(self.cfg.categories), "feature_extractor": report,
                   "vae_image_size": self.cfg.vae_image_size, "map_size": MAP,
                   "dataset": {"name": "MVTec AD", "mirror": MVTEC_HF_REPO, "revision": MVTEC_HF_REVISION}},
                  self.cfg.artifacts / "manifest.json")
        self.log(f"  feature extractor ONNX: {report}")

    # ------------------------------------------------------------ evaluate
    def stage_evaluate(self) -> None:
        costs = self.cfg.costs
        metric_rows, operating_rows, defect_rows, threshold_rows = [], [], [], []
        roc: dict = {}
        score_dump: dict = {}
        for category in self.cfg.categories:
            calibration = load_json(self.cfg.artifacts / category / "calibration.json")
            normalized = load_json(self.cfg.work / "scores" / f"normalized_{category}.json")
            test = self.paths(category, split="test")
            labels = test.label.to_numpy()
            operating = (test.role == "operating_test").to_numpy()
            masks = self.test_masks(category)
            maps_by_method = {}
            for method in ("patchcore", "vae", "mean"):
                data, _ = self._load_scores(method, category)
                maps_by_method[method] = normalize_map(data["test_maps"].astype(np.float32),
                                                       calibration["methods"][method]["pixel_scale"])
            w = calibration["methods"]["ensemble"]["weight_patchcore"]
            maps_by_method["ensemble"] = w * maps_by_method["patchcore"] + (1 - w) * maps_by_method["vae"]
            score_dump[category] = {"test_labels": labels.tolist(), "test_defect_types": test.defect_type.tolist(),
                                    "test_roles": test.role.tolist(), "methods": {}}
            for method in METHODS:
                scores = np.asarray(normalized[method]["test"])
                low, high = bootstrap_auroc_ci(labels, scores, n_boot=self.cfg.bootstrap, seed=self.cfg.seed)
                maps = maps_by_method[method]
                metric_rows.append({
                    "category": category, "method": method,
                    "image_auroc": safe_auroc(labels, scores), "image_auroc_ci_low": low, "image_auroc_ci_high": high,
                    "image_ap": safe_ap(labels, scores), "pixel_auroc": pixel_auroc(list(maps), masks),
                    "aupro_30": aupro(list(maps), masks, fpr_limit=0.3),
                    "synthetic_image_auroc": safe_auroc(
                        np.r_[np.zeros(len(normalized[method]["cal"])), np.ones(len(normalized[method]["syn"]))],
                        np.r_[normalized[method]["cal"], normalized[method]["syn"]]),
                    "test_images": int(len(labels)), "test_defects": int(labels.sum()),
                })
                roc.setdefault(category, {})[method] = roc_points(labels, scores)
                policies = calibration["methods"][method]["policies"]
                for name, policy in policies.items():
                    m = operating_metrics(scores[operating], labels[operating], policy["threshold"], policy["band"], costs)
                    operating_rows.append({"category": category, "method": method, "policy": name, **m})
                default = policies["synthetic_cost"]
                decisions = decide_many(scores, default["threshold"], default["band"])
                for defect_type in sorted(set(test.defect_type)):
                    mask = (test.defect_type == defect_type).to_numpy() & operating
                    if not mask.any():
                        continue
                    defect_rows.append({"category": category, "method": method, "defect_type": defect_type,
                                        "images": int(mask.sum()),
                                        "reject_rate": float(np.mean(decisions[mask] == "REJECT")),
                                        "check_rate": float(np.mean(decisions[mask] == "CHECK")),
                                        "pass_rate": float(np.mean(decisions[mask] == "PASS")),
                                        "mean_score": float(scores[mask].mean())})
                synthetic_t = policies["synthetic_cost"]
                real_t = policies["real_oracle_cost"]
                conspicuous_t = policies.get("synthetic_conspicuous_cost", synthetic_t)
                row = {"category": category, "method": method,
                       "conspicuous_threshold": conspicuous_t["threshold"],
                       "gap_conspicuous_vs_real": conspicuous_t["threshold"] - real_t["threshold"],
                       "synthetic_threshold": synthetic_t["threshold"], "synthetic_band": synthetic_t["band"],
                       "real_threshold": real_t["threshold"], "real_band": real_t["band"],
                       "good_quantile_threshold": policies["good_quantile"]["threshold"],
                       "gap_synthetic_vs_real": synthetic_t["threshold"] - real_t["threshold"]}
                if "diffusion_cost" in policies:
                    row["diffusion_threshold"] = policies["diffusion_cost"]["threshold"]
                    row["gap_diffusion_vs_real"] = policies["diffusion_cost"]["threshold"] - real_t["threshold"]
                    row["combined_threshold"] = policies["synthetic_plus_diffusion_cost"]["threshold"]
                    row["gap_combined_vs_real"] = policies["synthetic_plus_diffusion_cost"]["threshold"] - real_t["threshold"]
                threshold_rows.append(row)
                score_dump[category]["methods"][method] = {
                    k: np.round(np.asarray(v), 3).tolist() for k, v in normalized[method].items()}
            self._qualitative(category, test, maps_by_method, normalized, calibration)
        pd.DataFrame(metric_rows).to_csv(self.cfg.reports / "metrics.csv", index=False)
        pd.DataFrame(operating_rows).to_csv(self.cfg.reports / "operating_metrics.csv", index=False)
        pd.DataFrame(defect_rows).to_csv(self.cfg.reports / "per_defect_type.csv", index=False)
        pd.DataFrame(threshold_rows).to_csv(self.cfg.reports / "threshold_comparison.csv", index=False)
        save_json(roc, self.cfg.reports / "roc_curves.json")
        save_json(score_dump, self.cfg.reports / "scores.json")
        save_json({**costs.to_dict(), **reference_costs(costs)}, self.cfg.reports / "cost_model.json")
        self.log("\n" + pd.DataFrame(metric_rows)[["category", "method", "image_auroc", "pixel_auroc", "aupro_30"]]
                 .round(4).to_string(index=False))

    def _qualitative(self, category: str, test: pd.DataFrame, maps: dict, normalized: dict, calibration: dict) -> None:
        """Panel per defect type: input | ground truth | PatchCore | VAE | mean baseline."""
        from .viz import overlay

        picks = []
        scores = np.asarray(normalized["patchcore"]["test"])
        for defect_type, group in test.groupby("defect_type"):
            idx = group.index.to_numpy()
            # median-scoring example = representative, not cherry-picked
            picks.append((defect_type, int(idx[np.argsort(scores[idx])[len(idx) // 2]])))
        tile = 200
        header = 30
        columns = ["input", "ground truth", "PatchCore", "VAE", "mean baseline"]
        canvas = np.full((header + len(picks) * (tile + 22), len(columns) * tile, 3), 255, np.uint8)
        for c, title in enumerate(columns):
            cv2.putText(canvas, title, (c * tile + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 23, 42), 1, cv2.LINE_AA)
        for r, (defect_type, i) in enumerate(picks):
            record = test.iloc[i]
            image = resize_rgb(read_rgb(self.abs(record.image_path)), tile)
            mask = read_mask(self.abs(record.mask_path) if record.mask_path else None, tile)
            gt = image.copy()
            gt[mask > 0] = (0.5 * gt[mask > 0] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
            tiles = [image, gt] + [overlay(image, np.clip(maps[m][i], 0, 100)) for m in ("patchcore", "vae", "mean")]
            y = header + r * (tile + 22)
            for c, t in enumerate(tiles):
                canvas[y:y + tile, c * tile:(c + 1) * tile] = t
            label = f"{defect_type}  |  PatchCore score {scores[i]:.0f}"
            cv2.putText(canvas, label, (8, y + tile + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (51, 65, 85), 1, cv2.LINE_AA)
        Image.fromarray(canvas).save(self.cfg.figures / f"qualitative_{category}.jpg", quality=90)

    # ------------------------------------------------------------- latency
    def stage_latency(self) -> None:
        from .runtime import Inspector

        inspector = Inspector(self.cfg.artifacts)
        rows = []
        for category in self.cfg.categories:
            sample = [read_rgb(self.abs(p)) for p in self.paths(category, split="test").image_path[:6]]
            for method in inspector.methods(category):
                for image in sample[:2]:
                    inspector.inspect(image, category, method, compare=False)
                durations, breakdown = [], []
                for k in range(20):
                    result = inspector.inspect(sample[k % len(sample)], category, method, compare=False)
                    durations.append(result["timing"]["total_ms"])
                    breakdown.append(result["timing"])
                parts = pd.DataFrame(breakdown).median().to_dict()
                rows.append({"category": category, "method": method, "latency_ms_median": float(np.median(durations)),
                             "latency_ms_p95": float(np.quantile(durations, 0.95)),
                             "throughput_parts_per_min": 60000.0 / float(np.median(durations)),
                             **{f"stage_{k}": v for k, v in parts.items() if k != "total_ms"}})
        frame = pd.DataFrame(rows)
        frame["hardware"] = f"{platform.processor() or platform.machine()} CPU ({__import__('os').cpu_count()} threads)"
        frame.to_csv(self.cfg.reports / "latency.csv", index=False)
        self.log("\n" + frame[["category", "method", "latency_ms_median", "latency_ms_p95"]].round(1).to_string(index=False))

    # --------------------------------------------------------------- drift
    def stage_drift(self) -> None:
        from .runtime import Inspector

        inspector = Inspector(self.cfg.artifacts)
        rng = np.random.default_rng(self.cfg.seed)
        sens_rows, stream_rows, alarm_rows = [], [], []
        for category in self.cfg.categories:
            calibration = inspector.calibration(category)
            policy = inspector.policy(category, "patchcore")
            good_paths = list(self.paths(category, role="calibration_good").image_path) + \
                list(self.paths(category, split="test").query("label == 0").image_path)
            good = [read_rgb(self.abs(p)) for p in good_paths[:30]]
            defects = [read_rgb(self.abs(p)) for p in self.paths(category, split="test").query("label == 1")
                       .sample(n=12, random_state=self.cfg.seed).image_path]
            for kind in DRIFT_KINDS:
                for severity in (0.0, 0.25, 0.5, 0.75, 1.0):
                    res = [inspector.inspect(apply_drift(img, kind, severity, seed=i), category, compare=False,
                                             escalate_on_health=False) for i, img in enumerate(good)]
                    flagged = [any(f["status"] != "ok" for f in r["health"]) for r in res]
                    bad = [inspector.inspect(apply_drift(img, kind, severity, seed=i), category, compare=False,
                                             escalate_on_health=False) for i, img in enumerate(defects)]
                    sens_rows.append({
                        "category": category, "drift": kind, "severity": severity,
                        "good_mean_score": float(np.mean([r["score"] for r in res])),
                        "good_false_reject_rate": float(np.mean([r["decision"] == "REJECT" for r in res])),
                        "good_check_rate": float(np.mean([r["decision"] == "CHECK" for r in res])),
                        "health_flag_rate": float(np.mean(flagged)),
                        "defect_escape_rate": float(np.mean([r["decision"] == "PASS" for r in bad])),
                    })
            # Simulated production stream: qualified -> lamp ageing -> maintenance -> camera bumped.
            pool_good = good_paths
            pool_bad = list(self.paths(category, split="test").query("label == 1").image_path)
            ref_scores = np.asarray(load_json(self.cfg.work / "scores" / f"normalized_{category}.json")["patchcore"]["cal"])
            health_ref = calibration["health_reference"]
            records = []
            for t in range(1, 201):
                is_defect = rng.random() < 0.03
                path = pool_bad[rng.integers(len(pool_bad))] if is_defect else pool_good[rng.integers(len(pool_good))]
                image = read_rgb(self.abs(path))
                if t <= 60:
                    phase, kind, severity = "qualified", "none", 0.0
                elif t <= 120:
                    phase, kind, severity = "lamp ageing", "lighting_dim", 0.9 * (t - 60) / 60
                elif t <= 150:
                    phase, kind, severity = "after maintenance", "none", 0.0
                else:
                    phase, kind, severity = "camera bumped", "camera_shift", 0.55
                if kind != "none":
                    image = apply_drift(image, kind, severity, seed=t)
                r = inspector.inspect(image, category, compare=False, escalate_on_health=False)
                records.append({"category": category, "t": t, "phase": phase, "drift": kind, "severity": severity,
                                "is_defect": int(is_defect), "score": r["score"], "decision": r["decision"],
                                **{k: r["acquisition"][k] for k in ("brightness", "contrast", "sharpness")}})
            stream = pd.DataFrame(records)
            # Winsorise at the REJECT boundary: a single real defect is already handled by its REJECT
            # decision; the control chart watches the bulk of the score distribution.
            cap = policy["threshold"] + policy["band"]
            ewma = EWMAChart(float(ref_scores.mean()), float(ref_scores.std())).run(np.minimum(stream.score, cap))
            stream["score_ewma"], stream["score_ucl"], stream["score_alarm"] = ewma.ewma, ewma.ucl, ewma.alarm_high
            for signal in ("brightness", "sharpness"):
                chart = EWMAChart(health_ref[signal]["mean"], health_ref[signal]["std"]).run(stream[signal])
                stream[f"{signal}_ewma"], stream[f"{signal}_ucl"], stream[f"{signal}_lcl"] = chart.ewma, chart.ucl, chart.lcl
                stream[f"{signal}_alarm"] = chart.alarm_high | chart.alarm_low
                alarm_rows.append({"category": category, "signal": signal,
                                   "false_alarms_while_qualified": _alarm_count(chart, 0, 60),
                                   "first_alarm_after_lamp_ageing": first_alarm(chart, 60),
                                   "first_alarm_after_camera_bump": first_alarm(chart, 150)})
            alarm_rows.append({"category": category, "signal": "anomaly_score (winsorised)",
                               "false_alarms_while_qualified": _alarm_count(ewma, 0, 60),
                               "first_alarm_after_lamp_ageing": first_alarm(ewma, 60),
                               "first_alarm_after_camera_bump": first_alarm(ewma, 150)})
            stream_rows.append(stream)
            self.log(f"  drift {category}: done (policy t={policy['threshold']}, b={policy['band']})")
        pd.DataFrame(sens_rows).to_csv(self.cfg.reports / "drift_sensitivity.csv", index=False)
        pd.concat(stream_rows).to_csv(self.cfg.reports / "drift_stream.csv", index=False)
        pd.DataFrame(alarm_rows).to_csv(self.cfg.reports / "drift_alarms.csv", index=False)

    # ------------------------------------------------------------ few-shot
    def stage_few_shot(self) -> None:
        category = self.cfg.few_shot_category
        feats = self._features(category)
        test = self.paths(category, split="test")
        labels = test.label.to_numpy()
        masks = self.test_masks(category)
        rows = []
        n_train = len(feats["train"])
        config = PatchCoreConfig(backbone=self.cfg.backbone, image_size=self.cfg.image_size)
        costs = self.cfg.costs
        for shots in (1, 2, 4, 8, 16, 32, 64, n_train):
            repeats = 5 if shots <= 8 else (3 if shots < n_train else 1)
            for rep in range(repeats):
                rng = np.random.default_rng(self.cfg.seed + 97 * rep + shots)
                chosen = rng.choice(n_train, size=min(shots, n_train), replace=False)
                grids = feats["train"][chosen]
                n_vec = grids.shape[0] * grids.shape[1] * grids.shape[2]
                ratio = 1.0 if n_vec <= 20000 else max(0.1, 20000 / n_vec)
                bank = MemoryBank(build_memory_bank(grids, ratio, seed=self.cfg.seed))
                scores, maps = score_grids(feats["test"], bank, config)
                row = {"category": category, "shots": int(min(shots, n_train)), "repeat": rep,
                       "bank_vectors": len(bank), "image_auroc": safe_auroc(labels, scores),
                       "pixel_auroc": pixel_auroc(list(maps), masks)}
                # Label-free threshold exactly as in the app's onboarding: leave-one-image-out scores
                # of the k good shots; REJECT above the largest of them.
                if 2 <= shots <= 32:
                    flat = grids.reshape(-1, grids.shape[-1]).astype(np.float32)
                    owner = np.repeat(np.arange(len(grids)), grids.shape[1] * grids.shape[2])
                    loo = [score_grid(grids[i].astype(np.float32), MemoryBank(flat[owner != i]), config)[0]
                           for i in range(len(grids))]
                    t_raw = float(np.max(loo))
                    row.update({"false_reject_rate": float(np.mean(scores[labels == 0] >= t_raw)),
                                "escape_rate": float(np.mean(scores[labels == 1] < t_raw))})
                rows.append(row)
            self.log(f"  few-shot {shots}: AUROC={np.mean([r['image_auroc'] for r in rows if r['shots'] == min(shots, n_train)]):.3f}")
        pd.DataFrame(rows).to_csv(self.cfg.reports / "few_shot.csv", index=False)
        # Sample good + test images for live onboarding in the app.
        out = self.cfg.examples / "new_product" / category
        if out.exists():
            shutil.rmtree(out)
        (out / "good").mkdir(parents=True)
        (out / "test").mkdir(parents=True)
        for k, p in enumerate(self.paths(category, role="model_train_good").image_path[:10]):
            Image.fromarray(resize_rgb(read_rgb(self.abs(p)), 320)).save(out / "good" / f"good_{k:02d}.png")
        for defect_type, group in test.groupby("defect_type"):
            for k, p in enumerate(group.image_path[:2]):
                Image.fromarray(resize_rgb(read_rgb(self.abs(p)), 320)).save(out / "test" / f"{defect_type}_{k}.png")

    # ------------------------------------------------------------- examples
    def stage_examples(self) -> None:
        rows = []
        for category in self.cfg.categories:
            test = self.paths(category, split="test")
            folder = self.cfg.examples / category
            if folder.exists():
                shutil.rmtree(folder)
            (folder / "masks").mkdir(parents=True)
            picks = []
            for defect_type, group in test.groupby("defect_type"):
                picks.extend(group.sample(n=min(3 if defect_type == "good" else 2, len(group)),
                                          random_state=self.cfg.seed).itertuples())
            for record in picks:
                name = f"{record.defect_type}_{Path(record.image_path).stem}.png"
                Image.fromarray(resize_rgb(read_rgb(self.abs(record.image_path)), 448)).save(folder / name)
                mask_name = ""
                if record.mask_path:
                    mask_name = f"masks/{Path(name).stem}_mask.png"
                    Image.fromarray(read_mask(self.abs(record.mask_path), 448) * 255).save(folder / mask_name)
                rows.append({"category": category, "file": f"{category}/{name}", "mask": f"{category}/{mask_name}"
                             if mask_name else "", "label": int(record.label), "defect_type": record.defect_type,
                             "role": record.role, "source": record.image_path})
        pd.DataFrame(rows).to_csv(self.cfg.examples / "manifest.csv", index=False)
        (self.cfg.examples / "ATTRIBUTION.md").write_text(
            "# Example images\n\nDown-scaled test images from **MVTec AD** (Bergmann et al., IJCV 2021; CVPR 2019), "
            "redistributed under CC BY-NC-SA 4.0 for non-commercial research and teaching only.\n"
            "Original: https://www.mvtec.com/research-teaching/datasets/mvtec-ad\n", encoding="utf-8")

    # -------------------------------------------------------------- summary
    def stage_summary(self) -> None:
        sizes = []
        for path in sorted(self.cfg.artifacts.rglob("*")):
            if path.is_file():
                rel = path.relative_to(self.cfg.artifacts)
                sizes.append({"file": str(rel), "group": rel.parts[0] if len(rel.parts) > 1 else "shared",
                              "size_mb": path.stat().st_size / 1024 ** 2})
        pd.DataFrame(sizes).to_csv(self.cfg.reports / "artifact_sizes.csv", index=False)
        metrics = pd.read_csv(self.cfg.reports / "metrics.csv")
        operating = pd.read_csv(self.cfg.reports / "operating_metrics.csv")
        latency = pd.read_csv(self.cfg.reports / "latency.csv")
        thresholds = pd.read_csv(self.cfg.reports / "threshold_comparison.csv")
        macro = metrics.groupby("method")[["image_auroc", "pixel_auroc", "aupro_30"]].mean()
        default_ops = operating[operating.policy == "synthetic_cost"].groupby("method")[
            ["false_reject_rate", "escape_rate", "review_rate", "expected_cost_per_1000"]].mean()
        summary = {
            "categories": list(self.cfg.categories),
            "backbone": self.cfg.backbone, "image_size": self.cfg.image_size,
            "macro": macro.round(4).to_dict(orient="index"),
            "operating_default_policy": default_ops.round(4).to_dict(orient="index"),
            "latency_ms_median": latency.groupby("method").latency_ms_median.median().round(1).to_dict(),
            "mean_abs_threshold_gap": thresholds.assign(gap=thresholds.gap_synthetic_vs_real.abs())
            .groupby("method").gap.mean().round(2).to_dict(),
            "artifact_total_mb": round(float(pd.DataFrame(sizes).size_mb.sum()), 2),
            "feature_extractor_mb": round(float(load_json(self.cfg.artifacts / "manifest.json")
                                                ["feature_extractor"]["size_mb"]), 2),
            "costs": self.cfg.costs.to_dict(),
        }
        if (self.cfg.reports / "few_shot.csv").exists():
            few = pd.read_csv(self.cfg.reports / "few_shot.csv").groupby("shots").image_auroc.mean().round(4)
            summary["few_shot_auroc"] = {int(k): float(v) for k, v in few.items()}
        save_json(summary, self.cfg.reports / "summary.json")
        self.log(json.dumps(summary, indent=2)[:3000])
