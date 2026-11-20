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
from .decision import (CostModel, calibration_sets, default_policy, fit_pixel_scale, fit_score_scale,
                       normalize_map, normalize_scores,
                       operating_metrics, reference_costs, select_threshold_by_cost,
                       select_threshold_good_quantile, decide_many)
from .drift import DRIFT_KINDS, EWMAChart, apply_drift, first_alarm, health_check, reference_from_stats
from .metrics import aupro, bootstrap_auroc_ci, pixel_auroc, roc_points, safe_ap, safe_auroc
from .patchcore import (CoresetLadder, MemoryBank, PatchCoreConfig, build_memory_bank, patch_distances,
                        score_grid,
                        score_grids, scores_from_distances)
from .synthetic import generate_synthetic_set
from .fvae import FeatureVAEConfig, score_error_map
from .pose import align_image, align_mask, align_matrix, is_elongated, unalign_map
from .vae import VAEConfig, score_reconstruction

METHODS = ("patchcore", "fvae", "vae", "mean", "ensemble")
BASE_METHODS = ("patchcore", "fvae", "vae", "mean")  # detectors with their own model
MAP = 224  # common resolution for all evaluation maps
SEVERITY_LEVELS = (0.25, 0.5)  # reduced-strength replays of every synthetic defect (detection-limit set)


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
    # v2 search space (label-free selection, see stage_patchcore): input resolution, feature aggregation
    # (3x3 PatchCore neighbourhood vs raw descriptors) and rotated copies of good parts in the bank for
    # objects that arrive in any orientation (rotations keep thread handedness; flips would not).
    size_grid: tuple = (224, 288)
    pool_grid: tuple = (3, 1)
    object_rotation_grid: tuple = (1, 4)
    hpo_severity: float = 0.5  # subtle (50%-strength) synthetic defects join the selection objective
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
    fvae_epochs: int = 60
    stages: tuple = ("index", "eda", "synthetic", "patchcore", "vae", "fvae", "mean", "severity", "diffusion", "calibrate",
                     "evaluate", "requalify", "latency", "drift", "few_shot", "examples", "summary")

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
            for stale in [*(self.cfg.work / "features").glob(f"{category}_*_s[ye][nv].npz"),  # syn / sev caches
                          self.cfg.work / "scores" / f"patchcore_{category}_search.json",
                          self.cfg.work / "scores" / f"vae_{category}_done.json"]:
                stale.unlink(missing_ok=True)
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
    def _severity_items(self, category: str) -> tuple[list[tuple], np.ndarray]:
        """(source, synthetic, level) triples: each synthetic defect blended back into its good source."""
        items = [(row.source_path, row.image_path, level) for row in self.synthetic(category).itertuples()
                 for level in SEVERITY_LEVELS]
        return items, np.asarray([level for _, _, level in items], np.float32)

    @staticmethod
    def _blend(item: tuple) -> np.ndarray:
        source, defect, level = item
        source = read_rgb(source).astype(np.float32)
        defect = read_rgb(defect).astype(np.float32)
        return ((1 - level) * source + level * defect).clip(0, 255).astype(np.uint8)

    def _feature_items(self, category: str, name: str, rotations: int = 1, align: bool = False):
        items, augment = self._feature_items_raw(category, name, rotations)
        if not align:
            return items, augment
        if rotations > 1:
            raise ValueError("pose alignment and rotation augmentation are mutually exclusive")
        load = augment or (lambda item: [read_rgb(item)])
        return items, (lambda item: [align_image(view, category)[0] for view in load(item)])  # noqa: E731

    def _feature_items_raw(self, category: str, name: str, rotations: int = 1):
        augment = None
        if name == "train":
            items = [self.abs(p) for p in self.paths(category, role="model_train_good").image_path]
            if rotations > 1:
                augment = lambda path: [np.ascontiguousarray(np.rot90(read_rgb(path), k)) for k in range(rotations)]  # noqa: E731
        elif name == "cal":
            items = [self.abs(p) for p in self.paths(category, role="calibration_good").image_path]
        elif name == "test":
            items = [self.abs(p) for p in self.paths(category, split="test").image_path]
        elif name == "syn":
            items = self.synthetic(category).image_path.tolist()
        elif name == "sev":
            items, _ = self._severity_items(category)
            augment = lambda item: [self._blend(item)]  # noqa: E731
        else:
            raise ValueError(name)
        return items, augment

    def _feature_set(self, category: str, name: str, image_size: int | None = None,
                     rotations: int = 1, align: bool = False) -> dict[int, np.ndarray]:
        """Cached CNN patch features {pool_size: (N,h,w,C) float16} for one image set at one resolution."""
        from .features import EXPORT_POOLS, extract_features

        image_size = int(image_size or self.cfg.image_size)
        tag = (f"{name}_rot{rotations}" if rotations > 1 else name) + ("_aligned" if align else "")
        cache = self.cfg.work / "features" / f"{category}_{self.cfg.backbone}_{image_size}_{tag}.npz"
        if cache.exists():
            with np.load(cache) as data:
                return {int(k[1:]): data[k] for k in data.files}
        items, augment = self._feature_items(category, name, rotations, align)
        self.log(f"  features {category}/{tag} @{image_size}px: {len(items)} images")
        arrays = extract_features(self.extractor, items, image_size, device=self.cfg.device, pools=EXPORT_POOLS,
                                  augment=augment)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, **{f"p{p}": a for p, a in arrays.items()})
        return arrays

    def _elongated(self, category: str) -> bool:
        key = f"_elongated_{category}"
        if not hasattr(self, key):
            sample = [read_rgb(self.abs(p)) for p in self.paths(category, role="model_train_good").image_path[:12]]
            setattr(self, key, CATEGORY_INFO.get(category, {}).get("kind") == "object" and is_elongated(sample, category))
        return getattr(self, key)

    def _align_grid(self, category: str) -> tuple:
        return (False, True) if self._elongated(category) else (False,)

    def _pose_matrices(self, category: str, name: str) -> list:
        """Per-image (original->aligned affine, image shape) for an image set (same order as ``_feature_items``)."""
        cache = getattr(self, "_pose_cache", {})
        self._pose_cache = cache
        if (category, name) not in cache:
            items, augment = self._feature_items_raw(category, name)
            out = []
            for item in items:
                view = augment(item)[0] if augment else read_rgb(item)
                out.append((align_matrix(view, category), view.shape))
            cache[(category, name)] = out
        return cache[(category, name)]

    def _unalign_maps(self, category: str, name: str, maps: np.ndarray) -> np.ndarray:
        return np.stack([unalign_map(m, mat, shape, MAP)
                         for m, (mat, shape) in zip(maps, self._pose_matrices(category, name))]).astype(np.float32)

    def _rotation_grid(self, category: str) -> tuple:
        return self.cfg.object_rotation_grid if CATEGORY_INFO.get(category, {}).get("kind") == "object" else (1,)

    def stage_patchcore(self) -> None:
        """Label-free model selection on held-out good parts + synthetic defects (no real defect is used).

        Objective = 0.4 * image AUROC + 0.3 * TPR at <=1% false rejects + 0.3 * pixel AUROC, where the
        defect set is the full-strength synthetic defects *plus* their 50%-strength replays. Full-strength
        synthetic defects alone saturate (AUROC ~1) and favour settings that only see conspicuous blobs;
        the subtle replays reward settings that keep fine detail, which is what real defects need.
        """
        hpo_rows = []
        scoring = [("max", 0.01)] if "max" in self.cfg.score_modes else []
        if "topk" in self.cfg.score_modes:
            scoring += [("topk", f) for f in self.cfg.topk_grid]
        for category in self.cfg.categories:
            # Resumable per product: a finished search with the same search space is not repeated.
            base_space = {"backbone": self.cfg.backbone, "sizes": list(self.cfg.size_grid),
                                    "pools": list(self.cfg.pool_grid), "rotations": list(self._rotation_grid(category)),
                                    "coreset": list(self.cfg.coreset_grid), "blur": list(self.cfg.blur_grid),
                                    "scoring": [list(x) for x in scoring], "hpo_severity": self.cfg.hpo_severity,
                          "max_bank_mb": self.cfg.max_bank_mb, "seed": self.cfg.seed}
            signature = json.dumps({**base_space, **({"align": [False, True]} if self._elongated(category) else {})},
                                   sort_keys=True)
            unaligned_signature = json.dumps(base_space, sort_keys=True)
            done = self.cfg.work / "scores" / f"patchcore_{category}_search.json"
            previous = load_json(done, {}) if done.exists() else {}
            if previous.get("signature") == signature:
                rows = previous["rows"]
                self.log(f"  {category}: search already finished with this search space - reusing it")
            elif previous.get("signature") == unaligned_signature and self._elongated(category):
                # the unaligned part of the space was searched before (identical computation): add aligned only
                self.log(f"  {category}: reusing the unaligned search, adding pose-normalised configurations")
                rows = [{**r, "align": False} for r in previous["rows"]] + self._patchcore_search(category, scoring, (True,))
                previous = {"signature": signature, "rows": rows}
                save_json(previous, done)
            else:
                rows = self._patchcore_search(category, scoring)
                previous = {"signature": signature, "rows": rows}
                save_json(previous, done)
            config = self._select_patchcore(rows)
            for row in rows:
                row.setdefault("align", False)
                row["selected"] = all(row[k] == v or (k == "topk_fraction" and row["score_mode"] == "max")
                                      for k, v in self._row_key(config).items())
            hpo_rows.extend(rows)
            if (previous.get("selected") == config.to_dict()
                    and (self.cfg.artifacts / category / "patchcore_bank.npy").exists()
                    and (self.cfg.work / "scores" / f"patchcore_{category}.npz").exists()):
                self.log(f"  {category}: selected recipe unchanged - keeping its bank and scores")
                continue
            self._build_patchcore(category, config)
            save_json({**previous, "selected": config.to_dict()}, done)
        pd.DataFrame(hpo_rows).to_csv(self.cfg.reports / "hpo_patchcore.csv", index=False)

    # Objective differences below this are sampling noise: the standard error of an AUROC-type metric on ~50
    # held-out good vs ~250 synthetic parts is ~0.03. (Revised from 0.001 after the first full run: carpet's
    # 288 px recipe beat 224 px by 0.003 on synthetic data but was far more sensitive to the lot change between
    # MVTec's train and test captures - see README, "How the v2 recipe was chosen".)
    PARSIMONY_TOLERANCE = 0.005

    @staticmethod
    def _row_key(config: PatchCoreConfig) -> dict:
        return {"image_size": config.image_size, "pool_size": config.pool_size, "bank_rotations": config.bank_rotations,
                "coreset_ratio": config.coreset_ratio, "blur_sigma": config.blur_sigma, "score_mode": config.score_mode,
                "topk_fraction": config.topk_fraction, "align": config.align}

    def _select_patchcore(self, rows: list[dict]) -> PatchCoreConfig:
        """Parsimony rule: among configurations within PARSIMONY_TOLERANCE of the best label-free objective,
        deploy the one with the cheapest nearest-neighbour search (patches per image x bank vectors).

        On a saturated product (bottle: every option within 0.0004) a strict arg-max picks the most
        expensive recipe for no real gain; real improvements (screw: 288 px raw descriptors, +0.006 over the
        next cheaper recipe and +0.03 over v1's) are larger.
        """
        frame = pd.DataFrame(rows)
        frame["align"] = frame["align"].fillna(False).astype(bool) if "align" in frame else False
        near = frame[frame.objective >= frame.objective.max() - self.PARSIMONY_TOLERANCE].copy()
        near["nn_cost"] = (near.image_size / 8.0) ** 2 * near.bank_vectors
        choice = near.sort_values(["nn_cost", "objective"], ascending=[True, False]).iloc[0]
        best = frame.loc[frame.objective.idxmax()]
        config = PatchCoreConfig(backbone=self.cfg.backbone, image_size=int(choice.image_size),
                                 coreset_ratio=float(choice.coreset_ratio), blur_sigma=float(choice.blur_sigma),
                                 score_mode=str(choice.score_mode),
                                 topk_fraction=float(choice.topk_fraction) if choice.score_mode == "topk" else 0.01,
                                 max_bank_mb=self.cfg.max_bank_mb, pool_size=int(choice.pool_size),
                                 bank_rotations=int(choice.bank_rotations), align=bool(choice["align"]))
        self.log(f"  best objective {best.objective:.4f} ({int(best.image_size)}px pool={int(best.pool_size)} "
                 f"rot={int(best.bank_rotations)} align={bool(best["align"])} coreset={best.coreset_ratio}); deployed (parsimony, objective "
                 f"{choice.objective:.4f}, {int(choice.bank_vectors)} bank vectors): {config}")
        return config

    def _build_patchcore(self, category: str, config: PatchCoreConfig) -> None:
        """Build the deployed bank for the selected recipe and score every evaluation set with it."""
        _, sev_levels = self._severity_items(category)
        train = self._feature_set(category, "train", config.image_size, config.bank_rotations,
                                  config.align)[config.pool_size]
        bank_vectors = CoresetLadder(train, config.coreset_ratio, self.cfg.max_bank_mb, seed=self.cfg.seed).bank(
            config.coreset_ratio)
        folder = self.cfg.artifacts / category
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / "patchcore_bank.npy", bank_vectors)
        bank = MemoryBank(bank_vectors)
        payload = {"config": config.to_dict()}
        for name in ("cal", "syn", "test", "sev"):
            feats = self._feature_set(category, name, config.image_size, align=config.align)[config.pool_size]
            scores, maps = scores_from_distances(patch_distances(feats, bank), config, MAP)
            payload[f"{name}_scores"] = scores
            if name != "sev":
                if config.align:  # heat maps back to the original image frame
                    maps = self._unalign_maps(category, name, maps)
                payload[f"{name}_maps"] = maps.astype(np.float16)
        payload["sev_levels"] = sev_levels
        train1 = self._feature_set(category, "train", config.image_size, align=config.align)[config.pool_size][:24]
        payload["train_insample_scores"] = scores_from_distances(patch_distances(train1, bank), config)[0]
        self._save_scores("patchcore", category, payload)
        self.log(f"  {category}: deployed bank {len(bank_vectors)} vectors ({bank_vectors.nbytes / 1024 ** 2:.1f} MB)")

    def _patchcore_search(self, category: str, scoring: list, aligns: tuple | None = None) -> list[dict]:
        hpo_rows: list[dict] = []
        syn_masks = [read_mask(p, MAP) for p in self.synthetic(category).mask_path]
        _, sev_levels = self._severity_items(category)
        subtle = np.isclose(sev_levels, self.cfg.hpo_severity)
        best = None
        for size, align in [(sz, al) for sz in self.cfg.size_grid for al in (aligns or self._align_grid(category))]:
            cal = self._feature_set(category, "cal", size, align=align)
            syn = self._feature_set(category, "syn", size, align=align)
            sev = self._feature_set(category, "sev", size, align=align)
            if align:  # pixel AUROC in the aligned frame: warp the synthetic masks there
                masks = [align_mask(read_mask(p), mat, MAP)
                         for p, (mat, _) in zip(self.synthetic(category).mask_path, self._pose_matrices(category, "syn"))]
            else:
                masks = syn_masks
            for rotations in (self._rotation_grid(category) if not align else (1,)):
                train = self._feature_set(category, "train", size, rotations, align)
                for pool in self.cfg.pool_grid:
                    t0 = time.time()
                    ladder = CoresetLadder(train[pool], max(self.cfg.coreset_grid), self.cfg.max_bank_mb,
                                           seed=self.cfg.seed)
                    self.log(f"  {category} @{size}px align={align} pool={pool} rot={rotations}: greedy coreset of "
                             f"{len(ladder.order)} from {len(ladder.flat)} candidates / {ladder.n_total} patches "
                             f"({time.time() - t0:.0f}s)")
                    for ratio in self.cfg.coreset_grid:
                        t0 = time.time()
                        bank_vectors = ladder.bank(ratio)
                        bank = MemoryBank(bank_vectors)
                        d_cal = patch_distances(cal[pool], bank)
                        d_syn = patch_distances(syn[pool], bank)
                        d_sev = patch_distances(sev[pool][subtle], bank)
                        pixel_cache: dict = {}
                        for blur in self.cfg.blur_grid:
                            for mode, topk in scoring:
                                config = PatchCoreConfig(backbone=self.cfg.backbone, image_size=int(size),
                                                         coreset_ratio=ratio, blur_sigma=blur, score_mode=mode,
                                                         topk_fraction=topk, max_bank_mb=self.cfg.max_bank_mb,
                                                         pool_size=int(pool), bank_rotations=int(rotations),
                                                         align=bool(align))
                                good_s, good_m = scores_from_distances(d_cal, config, MAP)
                                syn_s, syn_m = scores_from_distances(d_syn, config, MAP)
                                sev_s, _ = scores_from_distances(d_sev, config)
                                defects = np.r_[syn_s, sev_s]
                                labels = np.r_[np.zeros(len(good_s)), np.ones(len(defects))]
                                image_auc = safe_auroc(labels, np.r_[good_s, defects])
                                # AUROC saturates on synthetic data; the share of defects caught above the
                                # 99th percentile of good scores (TPR at <=1% false rejects) does not, and it
                                # is the operating requirement of a real line.
                                tpr_at_1pct = float(np.mean(defects > np.quantile(good_s, 0.99)))
                                if blur not in pixel_cache:  # maps do not depend on the image-score mode
                                    pixel_cache[blur] = pixel_auroc(
                                        [*good_m, *syn_m], [np.zeros((MAP, MAP), np.uint8)] * len(good_m) + masks)
                                pix_auc = pixel_cache[blur]
                                bank_mb = len(bank) * bank.bank.shape[1] * 2 / 1024 ** 2
                                objective = (0.4 * image_auc + 0.3 * tpr_at_1pct + 0.3 * pix_auc
                                             - 1e-4 * np.log1p(bank_mb) - 1e-4 * (size / 224 - 1))
                                hpo_rows.append({
                                    "category": category, "image_size": int(size), "pool_size": int(pool),
                                    "bank_rotations": int(rotations), "align": bool(align),
                                    "coreset_ratio": ratio, "blur_sigma": blur,
                                    "score_mode": mode, "topk_fraction": topk if mode == "topk" else None,
                                    "bank_vectors": len(bank), "bank_mb": bank_mb,
                                    "synthetic_image_auroc": safe_auroc(
                                        np.r_[np.zeros(len(good_s)), np.ones(len(syn_s))], np.r_[good_s, syn_s]),
                                    "subtle_image_auroc": safe_auroc(
                                        np.r_[np.zeros(len(good_s)), np.ones(len(sev_s))], np.r_[good_s, sev_s]),
                                    "combined_image_auroc": image_auc, "synthetic_tpr_at_1pct_fpr": tpr_at_1pct,
                                    "synthetic_pixel_auroc": pix_auc, "objective": objective})
                                if best is None or objective > best[0]:
                                    best = (objective, config, bank_vectors)
                        self.log(f"  {category} @{size}px align={align} pool={pool} rot={rotations} coreset={ratio}: "
                                 f"bank={len(bank)} best objective so far={best[0]:.4f} ({time.time() - t0:.0f}s)")
                    del ladder
        return hpo_rows

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
            # Resumable per product: skip a model already trained with the same configuration.
            done = self.cfg.work / "scores" / f"vae_{category}_done.json"
            signature = json.dumps({**config.to_dict(), "seed": self.cfg.seed, "n_train": len(train)}, sort_keys=True)
            if (done.exists() and load_json(done, {}).get("signature") == signature
                    and (self.cfg.artifacts / category / "vae.onnx").exists()
                    and (self.cfg.work / "scores" / f"vae_{category}.npz").exists()):
                rows.extend(load_json(done)["history"])
                self.log(f"  {category}: VAE already trained with this configuration - reusing it")
                continue
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
            save_json({"signature": signature, "history": [{"category": category, **r} for r in config.history]}, done)
        pd.DataFrame(rows).to_csv(self.cfg.reports / "vae_training_history.csv", index=False)

    # ------------------------------------------------------- feature-space VAE
    def _fvae_config(self, category: str) -> FeatureVAEConfig:
        return FeatureVAEConfig(epochs=self.cfg.fvae_epochs, hflip=category != "screw")

    def _fvae_scores(self, session, feats: np.ndarray, config: FeatureVAEConfig) -> tuple[np.ndarray, np.ndarray]:
        scores, maps = [], []
        x = np.ascontiguousarray(np.asarray(feats, np.float32).transpose(0, 3, 1, 2))
        for start in range(0, len(x), 32):
            for err in session.run(None, {"features": x[start:start + 32]})[0]:
                score, amap = score_error_map(err, config, MAP)
                scores.append(score)
                maps.append(amap)
        return np.asarray(scores, np.float32), np.stack(maps).astype(np.float32)

    def stage_fvae(self) -> None:
        """Feature-space VAE (the "better VAE"): trained on good parts' CNN feature maps (224 px, 3x3 pooled)."""
        import onnxruntime as ort

        from .fvae import export_fvae_onnx, train_fvae

        rows = []
        for category in self.cfg.categories:
            config = self._fvae_config(category)
            train = self._feature_set(category, "train", config.image_size)[config.pool_size]
            done = self.cfg.work / "scores" / f"fvae_{category}_done.json"
            signature = json.dumps({**config.to_dict(), "seed": self.cfg.seed, "n_train": int(len(train))}, sort_keys=True)
            folder = self.cfg.artifacts / category
            if (done.exists() and load_json(done, {}).get("signature") == signature and (folder / "fvae.onnx").exists()
                    and (self.cfg.work / "scores" / f"fvae_{category}.npz").exists()):
                rows.extend(load_json(done)["history"])
                self.log(f"  {category}: feature-space VAE already trained with this configuration - reusing it")
                continue
            self.log(f"  training feature-space VAE for {category} on {len(train)} good feature maps")
            model, mean, std = train_fvae(train, config, seed=self.cfg.seed, device=self.cfg.device,
                                          progress=lambda r: self.log(f"    epoch {r['epoch']:3d} train={r['train_loss']:.4f} "
                                                                      f"val={r['val_loss']:.4f}")
                                          if r["epoch"] % 10 == 0 or r["epoch"] == 1 else None)
            report = export_fvae_onnx(model, mean, std, config, folder / "fvae.onnx", examples=train[:4])
            self.log(f"  exported {report}")
            session = ort.InferenceSession(str(folder / "fvae.onnx"), providers=["CPUExecutionProvider"])
            payload = {"config": {**config.to_dict(), "parameters": report["parameters"]}}
            for name in ("cal", "syn", "test", "sev"):
                feats = self._feature_set(category, name, config.image_size)[config.pool_size]
                scores, maps = self._fvae_scores(session, feats, config)
                payload[f"{name}_scores"] = scores
                if name != "sev":
                    payload[f"{name}_maps"] = maps.astype(np.float16)
            payload["sev_levels"] = self._severity_items(category)[1]
            self._save_scores("fvae", category, payload)
            history = [{"category": category, **r} for r in config.history]
            rows.extend(history)
            save_json({"signature": signature, "history": history}, done)
        pd.DataFrame(rows).to_csv(self.cfg.reports / "fvae_training_history.csv", index=False)

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
    def stage_severity(self) -> None:
        """Replay every synthetic defect at reduced severity (label-free 'detection-limit' set).

        Inspection specs define the faintest defect that must be caught. Blending each synthetic defect
        with its source good image at 25% / 50% strength spans subtle-to-obvious defects, so the cost-
        optimal threshold is not set by conspicuous synthetic defects only.
        """
        import onnxruntime as ort

        from .vae import load_vae_input

        for category in self.cfg.categories:
            items, levels = self._severity_items(category)
            data, config = self._load_scores("patchcore", category)
            pc_config = PatchCoreConfig(**config)
            bank = MemoryBank(np.load(self.cfg.artifacts / category / "patchcore_bank.npy"))
            grids = self._feature_set(category, "sev", pc_config.image_size, align=pc_config.align)[pc_config.pool_size]
            scores, _ = scores_from_distances(patch_distances(grids, bank), pc_config)
            self._save_scores("patchcore", category, {**data, "config": config, "sev_scores": scores,
                                                       "sev_levels": levels})
            data, config = self._load_scores("vae", category)
            vae_config = VAEConfig(**{k: v for k, v in config.items() if k in VAEConfig.__dataclass_fields__})
            session = ort.InferenceSession(str(self.cfg.artifacts / category / "vae.onnx"),
                                           providers=["CPUExecutionProvider"])
            model = MeanImageBaseline.load(self.cfg.artifacts / category)
            vae_scores, mean_scores = [], []
            for item in items:
                image = self._blend(item)
                x = load_vae_input(image, vae_config.image_size)
                vae_scores.append(score_reconstruction(x, session.run(None, {"image": x[None]})[0][0], vae_config)[0])
                mean_scores.append(model.predict(image)[0])
            self._save_scores("vae", category, {**data, "config": config,
                                                 "sev_scores": np.asarray(vae_scores, np.float32), "sev_levels": levels})
            data, config = self._load_scores("mean", category)
            self._save_scores("mean", category, {**data, "config": config,
                                                  "sev_scores": np.asarray(mean_scores, np.float32), "sev_levels": levels})
            if (self.cfg.artifacts / category / "fvae.onnx").exists():
                import onnxruntime as _ort
                data, config = self._load_scores("fvae", category)
                fcfg = FeatureVAEConfig(**{k: v for k, v in config.items() if k in FeatureVAEConfig.__dataclass_fields__})
                session = _ort.InferenceSession(str(self.cfg.artifacts / category / "fvae.onnx"), providers=["CPUExecutionProvider"])
                sev_scores, _ = self._fvae_scores(session, self._feature_set(category, "sev", fcfg.image_size)[fcfg.pool_size], fcfg)
                self._save_scores("fvae", category, {**data, "config": config, "sev_scores": sev_scores, "sev_levels": levels})
            self.log(f"  {category}: scored {len(items)} reduced-severity synthetic defects (levels {SEVERITY_LEVELS})")

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
            pc_config = PatchCoreConfig(**config)
            bank = MemoryBank(np.load(self.cfg.artifacts / category / "patchcore_bank.npy"))
            augment = (lambda item: [align_image(read_rgb(item), category)[0]]) if pc_config.align else None  # noqa: E731
            grids = extract_features(self.extractor, items, pc_config.image_size, device=self.cfg.device,
                                     pools=[pc_config.pool_size], augment=augment)[pc_config.pool_size]
            scores, maps = score_grids(grids, bank, pc_config, map_size=MAP)
            if pc_config.align:
                poses = [(align_matrix(img, category), img.shape) for img in map(read_rgb, items)]
                maps = np.stack([unalign_map(m, mat, shape, MAP) for m, (mat, shape) in zip(maps, poses)])
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
            if (self.cfg.artifacts / category / "fvae.onnx").exists():
                data, config = self._load_scores("fvae", category)
                fcfg = FeatureVAEConfig(**{k: v for k, v in config.items() if k in FeatureVAEConfig.__dataclass_fields__})
                session = ort.InferenceSession(str(self.cfg.artifacts / category / "fvae.onnx"), providers=["CPUExecutionProvider"])
                grids = extract_features(self.extractor, items, fcfg.image_size, device=self.cfg.device,
                                         pools=[fcfg.pool_size])[fcfg.pool_size]
                scores, maps = self._fvae_scores(session, grids, fcfg)
                self._save_scores("fvae", category, {**data, "config": config, "diff_scores": scores,
                                                      "diff_maps": maps.astype(np.float16)})
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
            for method in [m for m in BASE_METHODS if (self.cfg.work / "scores" / f"{m}_{category}.npz").exists()]:
                data, config = self._load_scores(method, category)
                scale = fit_score_scale(data["cal_scores"], data["syn_scores"])
                pixel_scale = fit_pixel_scale(data["cal_maps"].astype(np.float32),
                                              synthetic_maps=data["syn_maps"].astype(np.float32),
                                              synthetic_masks=syn_masks)
                normalized[method] = {k[:-7]: normalize_scores(v, scale) for k, v in data.items() if k.endswith("_scores")}
                methods[method] = {"config": config, "score_scale": scale, "pixel_scale": pixel_scale}
                if "sev_levels" in data:
                    sev_levels = data["sev_levels"]
            # Ensemble weight chosen label-free on held-out good vs full-strength + subtle (50%) synthetic
            # defects - the same defect set as the PatchCore model selection. Full-strength synthetics alone
            # reward the VAE for spotting conspicuous blobs and pulled the screw ensemble far below PatchCore.
            # Weights are scanned from PatchCore-only downwards, so ties keep the stronger single detector.
            subtle = (np.isclose(sev_levels, self.cfg.hpo_severity) if sev_levels is not None
                      else np.zeros(len(normalized["patchcore"].get("sev", [])), bool))
            partner = "fvae" if "fvae" in normalized else "vae"  # the generative partner of the ensemble
            best_w, best_auc = 1.0, -1.0
            for w in np.round(np.linspace(1, 0, 11), 2):
                mix = {k: w * normalized["patchcore"][k] + (1 - w) * normalized[partner][k] for k in ("cal", "syn", "sev")
                       if k in normalized["patchcore"] and k in normalized[partner]}
                defects = np.r_[mix["syn"], mix["sev"][subtle]] if "sev" in mix else mix["syn"]
                auc = safe_auroc(np.r_[np.zeros(len(mix["cal"])), np.ones(len(defects))], np.r_[mix["cal"], defects])
                if auc > best_auc + 1e-3:
                    best_w, best_auc = float(w), auc
            combined = {k: best_w * normalized["patchcore"][k] + (1 - best_w) * normalized[partner][k]
                        for k in normalized["patchcore"] if k in normalized[partner]}
            ens_scale = fit_score_scale(combined["cal"], combined["syn"])
            normalized["ensemble"] = {k: normalize_scores(v, ens_scale) for k, v in combined.items()}
            methods["ensemble"] = {"weight_patchcore": best_w, "partner": partner, "synthetic_auroc": best_auc,
                                   "score_scale": ens_scale,
                                   "pixel_scale": {"low": 0.0, "threshold": 50.0, "high": 100.0}, "config": {}}
            for method, info in methods.items():
                s = normalized[method]
                syn_labels = np.r_[np.zeros(len(s["cal"])), np.ones(len(s["syn"]))]
                groups = {**s, "sev_levels": sev_levels}
                # Label-free capability gate (decision.calibration_sets): a synthetic set joins the default
                # policy only if the detector separates it from held-out good parts (AUROC >= 0.9). Weaker
                # sets mark the detection limit and are reported instead of driving the cost optimum to
                # "reject everything". The same function drives the app's "apply cost-optimal" button.
                _, report = calibration_sets(groups)
                capability = {}
                for name, values, auc, used in report:
                    capability[name] = {"synthetic_auroc": auc, "used_for_default_threshold": used, "n": int(len(values))}
                    capability_rows.append({"category": category, "method": method, "calibration_set": name,
                                            "synthetic_auroc": auc, "n": int(len(values)), "used": used})
                info["detection_capability"] = capability
                policies = {
                    "synthetic_cost": default_policy(groups, costs),
                    "synthetic_conspicuous_cost": select_threshold_by_cost(np.r_[s["cal"], s["syn"]], syn_labels, costs),
                    "good_quantile": {"threshold": select_threshold_good_quantile(s["cal"], 0.01, 5.0), "band": 5.0},
                    "real_oracle_cost": select_threshold_by_cost(s["test"][oracle], labels[oracle], costs, smooth=True),
                }
                if "diff" in s:
                    diff_labels = np.r_[np.zeros(len(s["cal"])), np.ones(len(s["diff"]))]
                    policies["diffusion_cost"] = select_threshold_by_cost(np.r_[s["cal"], s["diff"]], diff_labels,
                                                                          costs, smooth=True)
                    both = np.r_[s["syn"], s["diff"]]
                    policies["synthetic_plus_diffusion_cost"] = select_threshold_by_cost(
                        np.r_[s["cal"], both], np.r_[np.zeros(len(s["cal"])), np.ones(len(both))], costs, smooth=True)
                info["policies"] = {name: {"threshold": float(p["threshold"]), "band": float(p["band"])}
                                    for name, p in policies.items()}
                info["default_policy"] = "synthetic_cost"
                # held-out good scores of qualification: reference for the label-free lot-shift test
                info["good_reference"] = np.round(np.asarray(s["cal"], np.float64), 3).tolist()
                info["policy_notes"] = {
                    "synthetic_cost": "default: held-out good + every synthetic set the detector can reliably separate "
                                      "(severity 100/50/25% and diffusion-refined; synthetic AUROC >= 0.9); cost "
                                      "minimised on kernel-smoothed score distributions",
                    "synthetic_conspicuous_cost": "held-out good + full-strength synthetic defects only, raw counts "
                                                  "(first version)",
                    "good_quantile": "1% false-reject quantile of held-out good parts (no defects at all)",
                    "real_oracle_cost": "labelled real defects (oracle split, smoothed like the default) - comparison "
                                        "only, never deployed"}
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

        report = export_onnx(self.extractor.cpu(), self.cfg.artifacts / "feature_extractor.onnx", self.cfg.image_size,
                             check_sizes=self.cfg.size_grid)
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
            sev_levels = None
            for method in [m for m in BASE_METHODS if m in calibration["methods"]]:
                data, _ = self._load_scores(method, category)
                if sev_levels is None and "sev_levels" in data:
                    sev_levels = np.asarray(data["sev_levels"]).round(3).tolist()
                maps_by_method[method] = normalize_map(data["test_maps"].astype(np.float32),
                                                       calibration["methods"][method]["pixel_scale"])
            w = calibration["methods"]["ensemble"]["weight_patchcore"]
            partner = calibration["methods"]["ensemble"].get("partner", "vae")
            maps_by_method["ensemble"] = w * maps_by_method["patchcore"] + (1 - w) * maps_by_method[partner]
            score_dump[category] = {"test_labels": labels.tolist(), "test_defect_types": test.defect_type.tolist(),
                                    "test_roles": test.role.tolist(), "sev_levels": sev_levels, "methods": {}}
            for method in [m for m in METHODS if m in normalized]:
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

    # ------------------------------------------------------- lot re-qualification
    def stage_requalify(self) -> None:
        """Lot change: label-free shift test on GOOD parts of the current lot; re-qualify PatchCore where it fires.

        MVTec's test images were captured separately from the training images (carpet: a brighter, different roll).
        The oracle split's good parts play the "N good parts confirmed by inspectors from the current lot"; the
        operating split (disjoint) measures the effect. Only good parts are used - no defect labels.
        """
        from .runtime import Inspector

        inspector = Inspector(self.cfg.artifacts)
        rows, dists = [], {}
        costs = self.cfg.costs
        for category in self.cfg.categories:
            test = self.paths(category, split="test")
            new_goods = test[(test.role == "oracle_calibration") & (test.label == 0)]
            images = [read_rgb(self.abs(p)) for p in new_goods.image_path]
            lot = inspector.requalify(category, images, costs=costs)
            shift = lot["lot_shift"]
            operating = test[test.role == "operating_test"]
            labels = operating.label.to_numpy()
            normalized = load_json(self.cfg.work / "scores" / f"normalized_{category}.json")
            base_scores = np.asarray(normalized["patchcore"]["test"])[(test.role == "operating_test").to_numpy()]
            base_policy = inspector.policy(category, "patchcore")
            lot_scores = np.array([inspector._patchcore(read_rgb(self.abs(p)), category, lot)["score"]
                                   for p in operating.image_path])
            base = operating_metrics(base_scores, labels, base_policy["threshold"], base_policy["band"], costs)
            after = operating_metrics(lot_scores, labels, lot["threshold"], lot["band"], costs)
            row = {"category": category, "n_good_current_lot": shift["n_good"],
                   "current_lot_median": shift["current_lot_median"],
                   "qualification_median": shift["qualification_median"], "p_value": shift["p_value"],
                   "lot_shift": shift["shifted"], "deployed": shift["shifted"],
                   "base_threshold": base_policy["threshold"], "base_band": base_policy["band"],
                   "requal_threshold": lot["threshold"], "requal_band": lot["band"],
                   "base_auroc": safe_auroc(labels, base_scores), "requal_auroc": safe_auroc(labels, lot_scores)}
            for key in ("false_reject_rate", "escape_rate", "good_check_rate", "review_rate", "expected_cost_per_1000"):
                row[f"base_{key}"], row[f"requal_{key}"] = base[key], after[key]
            rows.append(row)
            dists[category] = {"current_lot_good_base": shift["current_lot_scores"], "good_loo": lot["good_scores"],
                               "synthetic": lot["synthetic_scores"], "threshold": lot["threshold"], "band": lot["band"],
                               "operating_labels": labels.astype(int).tolist(),
                               "operating_scores_base": np.round(base_scores, 3).tolist(),
                               "operating_scores_requalified": np.round(lot_scores, 3).tolist(),
                               "base_threshold": base_policy["threshold"], "base_band": base_policy["band"]}
            folder = self.cfg.artifacts / category / "requalified"
            if shift["shifted"]:
                folder.mkdir(parents=True, exist_ok=True)
                np.save(folder / "patchcore_bank.npy", lot["bank"])
                save_json({k: v for k, v in lot.items() if k not in ("bank", "synthetic_examples") and not k.startswith("_")}
                          | {"source": "good parts of the current lot (MVTec test split, oracle part)",
                             "created": time.strftime("%Y-%m-%d %H:%M:%S")}, folder / "requalification.json")
            elif folder.exists():
                shutil.rmtree(folder)
            sample = self.cfg.examples / "requalify" / category
            if sample.exists():
                shutil.rmtree(sample)
            sample.mkdir(parents=True)
            for k, image in enumerate(images):
                Image.fromarray(resize_rgb(image, 448)).save(sample / f"current_lot_good_{k:02d}.png")
            self.log(f"  {category}: lot shift p={shift['p_value']:.3g} ({'re-qualified' if shift['shifted'] else 'no shift'}); "
                     f"operating split FRR {base['false_reject_rate']:.3f}->{after['false_reject_rate']:.3f}, escapes "
                     f"{base['escape_rate']:.3f}->{after['escape_rate']:.3f}, good CHECK {base['good_check_rate']:.3f}->"
                     f"{after['good_check_rate']:.3f}")
        pd.DataFrame(rows).to_csv(self.cfg.reports / "requalification.csv", index=False)
        save_json(dists, self.cfg.reports / "requalification_scores.json")

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
        panel = [m for m in ("patchcore", "fvae", "vae", "mean") if m in maps]
        names = {"patchcore": "PatchCore", "fvae": "feature VAE", "vae": "pixel VAE", "mean": "mean baseline"}
        columns = ["input", "ground truth"] + [names[m] for m in panel]
        canvas = np.full((header + len(picks) * (tile + 22), len(columns) * tile, 3), 255, np.uint8)
        for c, title in enumerate(columns):
            cv2.putText(canvas, title, (c * tile + 8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (15, 23, 42), 1, cv2.LINE_AA)
        for r, (defect_type, i) in enumerate(picks):
            record = test.iloc[i]
            image = resize_rgb(read_rgb(self.abs(record.image_path)), tile)
            mask = read_mask(self.abs(record.mask_path) if record.mask_path else None, tile)
            gt = image.copy()
            gt[mask > 0] = (0.5 * gt[mask > 0] + 0.5 * np.array([255, 40, 40])).astype(np.uint8)
            tiles = [image, gt] + [overlay(image, np.clip(maps[m][i], 0, 100)) for m in panel]
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
        feats = {name: self._feature_set(category, name, self.cfg.image_size)[3] for name in ("train", "test")}
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
                scores, maps = score_grids(feats["test"], bank, config, map_size=MAP)
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
            "patchcore_configs": {c: {k: v for k, v in load_json(self.cfg.artifacts / c / "calibration.json")
                                      ["methods"]["patchcore"]["config"].items()
                                      if k in ("image_size", "align", "pool_size", "bank_rotations", "coreset_ratio",
                                               "blur_sigma", "score_mode", "topk_fraction")}
                                  for c in self.cfg.categories},
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
