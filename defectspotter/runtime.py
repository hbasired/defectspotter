"""Inference runtime used by the app, the API and the batch tools.

Dependencies: NumPy, OpenCV, ONNX Runtime (no PyTorch). Artifacts layout::

    artifacts/
      manifest.json                 versions, backbone, sizes, categories
      feature_extractor.onnx        shared frozen CNN (PatchCore features)
      <category>/
        calibration.json            scales, thresholds, configs, health reference
        patchcore_bank.npy          float16 memory bank (greedy coreset)
        vae.onnx                    convolutional VAE (reconstruction)
        mean_baseline.npz/.json     aligned mean-image baseline
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np
import pandas as pd

from .baseline import MeanImageBaseline
from .constants import DEFAULT_ARTIFACT_DIR
from .data import acquisition_stats, imagenet_input, load_json, to_rgb_array
from .decision import decide, fit_pixel_scale, fit_score_scale, normalize_map, normalize_scores
from .decision import CostModel, default_policy, select_threshold_by_cost
from .drift import health_check
from .patchcore import MemoryBank, PatchCoreConfig, build_memory_bank, greedy_coreset, score_grid
from .synthetic import make_synthetic_defect
from .fvae import FeatureVAEConfig, score_error_map
from .pose import align_image, align_mask, unalign_map
from .vae import VAEConfig, load_vae_input, score_reconstruction
from .viz import draw_regions, find_regions, fit_display, overlay

METHOD_ORDER = ("patchcore", "fvae", "vae", "mean", "ensemble")
MAP_SIZE = 224


def _session(path: Path):
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = max(1, min(8, os.cpu_count() or 2))
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _resize_map(anomaly_map: np.ndarray, size: int = MAP_SIZE) -> np.ndarray:
    if anomaly_map.shape[0] == size and anomaly_map.shape[1] == size:
        return anomaly_map.astype(np.float32)
    return cv2.resize(anomaly_map.astype(np.float32), (size, size), interpolation=cv2.INTER_LINEAR)


class Inspector:
    """Load trained artifacts lazily and inspect parts with any detector."""

    def __init__(self, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR) -> None:
        self.root = Path(artifact_dir)
        if not (self.root / "manifest.json").exists():
            raise FileNotFoundError(f"No manifest.json in {self.root}. Train first or download artifacts.")
        self.manifest = load_json(self.root / "manifest.json")
        self._lock = threading.RLock()
        self._extractor = None
        self._banks: dict[str, MemoryBank] = {}
        self._vaes: dict[str, object] = {}
        self._means: dict[str, MeanImageBaseline] = {}
        self._calibration: dict[str, dict] = {}

    # ------------------------------------------------------------ metadata
    @property
    def categories(self) -> list[str]:
        return [c for c in self.manifest.get("categories", []) if (self.root / c / "calibration.json").exists()]

    def calibration(self, category: str) -> dict:
        if category not in self._calibration:
            self._calibration[category] = load_json(self.root / category / "calibration.json")
        return self._calibration[category]

    def methods(self, category: str) -> list[str]:
        available = self.calibration(category).get("methods", {})
        return [m for m in METHOD_ORDER if m in available]

    def policy(self, category: str, method: str, name: str | None = None) -> dict:
        info = self.calibration(category)["methods"][method]
        name = name or info.get("default_policy", "synthetic_cost")
        policy = dict(info["policies"][name])
        policy["name"] = name
        return policy

    # ------------------------------------------------------------- loaders
    @property
    def extractor(self):
        with self._lock:
            if self._extractor is None:
                self._extractor = _session(self.root / self.manifest["feature_extractor"]["path"])
            return self._extractor

    def bank(self, category: str) -> MemoryBank:
        with self._lock:
            if category not in self._banks:
                self._banks[category] = MemoryBank(np.load(self.root / category / "patchcore_bank.npy"))
            return self._banks[category]

    def fvae_session(self, category: str):
        with self._lock:
            key = f"fvae:{category}"
            if key not in self._vaes:
                self._vaes[key] = _session(self.root / category / "fvae.onnx")
            return self._vaes[key]

    def vae(self, category: str):
        with self._lock:
            if category not in self._vaes:
                self._vaes[category] = _session(self.root / category / "vae.onnx")
            return self._vaes[category]

    def mean_model(self, category: str) -> MeanImageBaseline:
        with self._lock:
            if category not in self._means:
                self._means[category] = MeanImageBaseline.load(self.root / category)
            return self._means[category]

    def warmup(self) -> None:
        dummy = np.full((64, 64, 3), 127, np.uint8)
        for category in self.categories:
            for method in self.methods(category):
                self.inspect(dummy, category, method, compare=False)

    # ------------------------------------------------------------ features
    def patch_grid(self, rgb: np.ndarray, image_size: int, pool_size: int = 3) -> np.ndarray:
        """CNN patch descriptors (h, w, C). The shared ONNX graph has one output per pooling size."""
        x = imagenet_input(rgb, image_size)[None]
        outputs = self.manifest["feature_extractor"].get("outputs", {"3": None})
        name = outputs.get(str(int(pool_size)))
        if name is None and str(int(pool_size)) not in outputs:
            raise ValueError(f"The feature extractor has no output for pool_size={pool_size} ({outputs})")
        out = self.extractor.run([name] if name else None, {"image": x})[0][0]
        return np.ascontiguousarray(out.transpose(1, 2, 0))

    def pc_grid(self, rgb: np.ndarray, category: str, config: PatchCoreConfig):
        """PatchCore patch grid of one image (pose-normalised first for elongated parts) + the alignment affine."""
        if config.align:
            aligned, matrix = align_image(rgb, category)
            return self.patch_grid(aligned, config.image_size, config.pool_size), matrix
        return self.patch_grid(rgb, config.image_size, config.pool_size), None

    # --------------------------------------------------------- per method
    def _patchcore(self, rgb: np.ndarray, category: str, lot: dict | None = None) -> dict:
        info = self.calibration(category)["methods"]["patchcore"]
        config = PatchCoreConfig(**info["config"])
        bank = self.bank(category)
        score_scale, pixel_scale = info["score_scale"], info["pixel_scale"]
        if lot is not None:  # re-qualified for the current lot: updated bank + re-fitted scales
            bank = lot.get("_bank_cache") or MemoryBank(lot["bank"])
            lot["_bank_cache"] = bank
            score_scale, pixel_scale = lot["score_scale"], lot["pixel_scale"]
        t0 = time.perf_counter()
        grid, matrix = self.pc_grid(rgb, category, config)
        t1 = time.perf_counter()
        raw, anomaly_map, _ = score_grid(grid, bank, config)
        t2 = time.perf_counter()
        norm = normalize_map(anomaly_map, pixel_scale)
        amap = unalign_map(norm, matrix, rgb.shape, MAP_SIZE) if matrix is not None else _resize_map(norm)
        return {"raw": raw, "score": float(normalize_scores(raw, score_scale)), "map": amap,
                "timing": {"cnn_features_ms": 1000 * (t1 - t0), "nearest_neighbour_ms": 1000 * (t2 - t1)},
                "grid_mean": grid.reshape(-1, grid.shape[-1]).mean(0)}

    def _vae(self, rgb: np.ndarray, category: str) -> dict:
        info = self.calibration(category)["methods"]["vae"]
        config = VAEConfig(**info["config"])
        t0 = time.perf_counter()
        x = load_vae_input(rgb, config.image_size)
        recon = self.vae(category).run(None, {"image": x[None]})[0][0]
        raw, anomaly_map = score_reconstruction(x, recon, config)
        t1 = time.perf_counter()
        recon_rgb = (np.clip(recon.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)
        return {"raw": raw, "score": float(normalize_scores(raw, info["score_scale"])),
                "map": _resize_map(normalize_map(anomaly_map, info["pixel_scale"])),
                "timing": {"vae_ms": 1000 * (t1 - t0)}, "reconstruction": recon_rgb}

    def _fvae(self, rgb: np.ndarray, category: str) -> dict:
        info = self.calibration(category)["methods"]["fvae"]
        config = FeatureVAEConfig(**{k: v for k, v in info["config"].items() if k in FeatureVAEConfig.__dataclass_fields__})
        t0 = time.perf_counter()
        grid = self.patch_grid(rgb, config.image_size, config.pool_size)
        err = self.fvae_session(category).run(None, {"features": np.ascontiguousarray(grid.transpose(2, 0, 1))[None]})[0][0]
        raw, anomaly_map = score_error_map(err, config, MAP_SIZE)
        t1 = time.perf_counter()
        return {"raw": raw, "score": float(normalize_scores(raw, info["score_scale"])),
                "map": _resize_map(normalize_map(anomaly_map, info["pixel_scale"])),
                "timing": {"feature_vae_ms": 1000 * (t1 - t0)}}

    def _mean(self, rgb: np.ndarray, category: str) -> dict:
        info = self.calibration(category)["methods"]["mean"]
        t0 = time.perf_counter()
        raw, anomaly_map = self.mean_model(category).predict(rgb)
        t1 = time.perf_counter()
        return {"raw": raw, "score": float(normalize_scores(raw, info["score_scale"])),
                "map": _resize_map(normalize_map(anomaly_map, info["pixel_scale"])),
                "timing": {"mean_baseline_ms": 1000 * (t1 - t0)}}

    def _run_method(self, rgb: np.ndarray, category: str, method: str, cache: dict, lot: dict | None = None) -> dict:
        if method in cache:
            return cache[method]
        if method == "patchcore" and lot is not None:
            cache[method] = self._patchcore(rgb, category, lot)
            return cache[method]
        if method == "ensemble":
            info = self.calibration(category)["methods"]["ensemble"]
            weight = float(info["weight_patchcore"])
            p = self._run_method(rgb, category, "patchcore", cache)
            v = self._run_method(rgb, category, info.get("partner", "vae"), cache)
            raw = weight * p["score"] + (1 - weight) * v["score"]
            result = {"raw": raw, "score": float(normalize_scores(raw, info["score_scale"])),
                      "map": weight * p["map"] + (1 - weight) * v["map"],
                      "timing": {**p["timing"], **v["timing"]}, "reconstruction": v.get("reconstruction")}
        else:
            result = {"patchcore": self._patchcore, "fvae": self._fvae, "vae": self._vae,
                      "mean": self._mean}[method](rgb, category)
        cache[method] = result
        return result

    # ------------------------------------------------------------- public
    def inspect(self, image, category: str, method: str = "patchcore", threshold: float | None = None,
                band: float | None = None, compare: bool = True, escalate_on_health: bool = True,
                display_size: int = 512, lot: dict | None = None) -> dict:
        """Inspect one part. Returns decision, 0-100 score, maps, overlay, regions, health, timings.

        ``lot``: a re-qualified PatchCore model for the current production lot (see ``requalify``); it replaces the
        PatchCore bank, score scale and default threshold. Other detectors keep the qualification-lot models.
        """
        started = time.perf_counter()
        rgb = to_rgb_array(image)
        policy = self.lot_policy(lot) if (lot is not None and method == "patchcore") else self.policy(category, method)
        threshold = float(policy["threshold"] if threshold is None else threshold)
        band = float(policy["band"] if band is None else band)
        cache: dict[str, dict] = {}
        result = self._run_method(rgb, category, method, cache, lot)
        score = float(result["score"])
        decision = decide(score, threshold, band)

        stats = acquisition_stats(rgb)
        flags = health_check(stats, self.calibration(category).get("health_reference", {}))
        out_of_range = [f for f in flags if f["status"] != "ok"]
        reasons = []
        if out_of_range and escalate_on_health and decision != "CHECK":
            # No automatic decision on an unqualified image: a PASS could be an escape and a
            # REJECT is most likely a lighting/focus false reject. Route it to a human.
            reasons.append(f"Image acquisition outside the qualified range ({', '.join(f['signal'] for f in out_of_range)})"
                           f" - model said {decision}, but no automatic decision is taken on an unqualified image. "
                           "Check lamp / lens / camera.")
            decision = "CHECK"

        display = fit_display(rgb, display_size)
        map_display = np.clip(result["map"], 0, 100)
        regions = find_regions(map_display, level=50.0, out_size=(display.shape[1], display.shape[0]))
        heat_overlay = overlay(display, map_display)
        annotated = draw_regions(heat_overlay, regions) if decision != "PASS" else heat_overlay

        comparison = []
        if compare:
            for other in self.methods(category):
                try:
                    r = self._run_method(rgb, category, other, cache, lot)
                    pol = (self.lot_policy(lot) if (lot is not None and other == "patchcore")
                           else self.policy(category, other))
                    comparison.append({"method": other, "score": round(float(r["score"]), 1),
                                       "threshold": pol["threshold"], "band": pol["band"],
                                       "decision": decide(float(r["score"]), pol["threshold"], pol["band"]),
                                       "map": np.clip(r["map"], 0, 100)})
                except Exception as exc:  # keep the main result even if a secondary method fails
                    comparison.append({"method": other, "score": float("nan"), "decision": f"error: {exc}"})
        timing = dict(result["timing"])
        timing["total_ms"] = 1000 * (time.perf_counter() - started)
        reconstruction = result.get("reconstruction")
        if reconstruction is None and "vae" in cache:
            reconstruction = cache["vae"].get("reconstruction")
        return {
            "category": category, "method": method, "decision": decision, "score": score,
            "raw_score": float(result["raw"]), "threshold": threshold, "band": band, "policy": policy["name"],
            "anomaly_map": map_display, "overlay": annotated, "display": display,
            "regions": regions, "reconstruction": reconstruction,
            "health": flags, "health_reasons": reasons, "acquisition": stats,
            "comparison": comparison, "timing": timing,
        }

    def inspect_batch(self, images: Sequence, names: Sequence[str], category: str, method: str = "patchcore",
                      threshold: float | None = None, band: float | None = None,
                      lot: dict | None = None) -> tuple[pd.DataFrame, list]:
        rows, overlays = [], []
        for image, name in zip(images, names):
            r = self.inspect(image, category, method, threshold, band, compare=False, lot=lot)
            rows.append({"file": name, "decision": r["decision"], "score": round(r["score"], 1),
                         "threshold": r["threshold"], "band": r["band"], "regions": int(len(r["regions"])),
                         "largest_region_pct": float(r["regions"]["area_pct"].max()) if len(r["regions"]) else 0.0,
                         "health": "ok" if not r["health_reasons"] else "out of range",
                         "latency_ms": round(r["timing"]["total_ms"], 1)})
            overlays.append((r["overlay"], f"{name} · {r['decision']} · {r['score']:.0f}"))
        return pd.DataFrame(rows), overlays

    # ------------------------------------------------- lot re-qualification
    @staticmethod
    def lot_policy(lot: dict) -> dict:
        return {"threshold": float(lot["threshold"]), "band": float(lot["band"]),
                "name": "re-qualified (current lot)"}

    def shipped_lot(self, category: str) -> dict | None:
        """Re-qualified model shipped with the artifacts (pipeline stage ``requalify``), if a lot shift was found."""
        folder = self.root / category / "requalified"
        if not (folder / "requalification.json").exists():
            return None
        with self._lock:
            key = f"lot:{category}"
            if key not in self._calibration:
                lot = load_json(folder / "requalification.json")
                lot["bank"] = np.load(folder / "patchcore_bank.npy")
                self._calibration[key] = lot
            return self._calibration[key]

    def lot_shift_test(self, category: str, good_images: Sequence) -> dict:
        """Label-free lot-change test: do good parts from the current lot score higher than the held-out good
        parts of qualification? (one-sided Mann-Whitney U; only GOOD parts are needed)."""
        from scipy.stats import mannwhitneyu

        info = self.calibration(category)["methods"]["patchcore"]
        config = PatchCoreConfig(**info["config"])
        scores = np.array([score_grid(self.pc_grid(to_rgb_array(i), category, config)[0], self.bank(category), config)[0]
                           for i in good_images])
        new = normalize_scores(scores, info["score_scale"])
        reference = np.asarray(info.get("good_reference") or [], np.float64)
        p_value = float(mannwhitneyu(new, reference, alternative="greater").pvalue) if len(reference) and len(new) > 2 else float("nan")
        return {"n_good": int(len(new)), "current_lot_median": float(np.median(new)),
                "qualification_median": float(np.median(reference)) if len(reference) else float("nan"),
                "p_value": p_value, "shifted": bool(np.isfinite(p_value) and p_value < 0.01),
                "current_lot_scores": new.round(2).tolist()}

    def requalify(self, category: str, good_images: Sequence, coreset_ratio: float = 0.10,
                  synthetic_per_image: int = 4, seed: int = 7000, costs: CostModel | None = None) -> dict:
        """Re-qualify PatchCore for a new production lot from a handful of GOOD parts of that lot.

        1. memory bank += greedy coreset (10%) of the new lot's patches (old lots stay normal);
        2. good reference = leave-one-image-out scores of the new parts;
        3. synthetic defects (cut-paste / scratch / stain, + 50% replays) generated FROM the new parts;
        4. score scale (25 = typical new good part, 75 = typical synthetic defect) and the cost-optimal
           PASS / CHECK / REJECT thresholds are re-fitted exactly like the qualification default.
        No defect image is needed.
        """
        if len(good_images) < 3:
            raise ValueError("Re-qualification needs at least 3 good parts of the current lot (10+ recommended).")
        info = self.calibration(category)["methods"]["patchcore"]
        config = PatchCoreConfig(**info["config"])
        rgbs = [to_rgb_array(i) for i in good_images]
        grids = np.stack([self.pc_grid(rgb, category, config)[0] for rgb in rgbs])
        n, h, w, c = grids.shape
        shift = self.lot_shift_test(category, rgbs)
        base = self.bank(category).bank
        flat = grids.reshape(-1, c).astype(np.float32)
        owner = np.repeat(np.arange(n), h * w)
        order = greedy_coreset(flat, max(1, int(round(len(flat) * coreset_ratio))), seed=seed)
        bank = MemoryBank(np.vstack([base, flat[order]]))
        loo, good_maps = [], []
        for i in range(n):  # leave-one-image-out: bank + the other new parts' coreset patches
            raw, amap, _ = score_grid(grids[i], MemoryBank(np.vstack([base, flat[order[owner[order] != i]]])), config)
            loo.append(raw)
            good_maps.append(amap)
        syn, sev, syn_maps, syn_masks, examples = [], [], [], [], []
        size = config.image_size
        for i, rgb in enumerate(rgbs):
            for j in range(synthetic_per_image):
                defect, mask, kind = make_synthetic_defect(rgb, category, seed + 31 * i + j)
                grid, matrix = self.pc_grid(defect, category, config)
                raw, amap, _ = score_grid(grid, bank, config)
                syn.append(raw)
                syn_maps.append(amap)  # maps and masks both in the (possibly aligned) feature frame
                syn_masks.append(align_mask(mask, matrix, size) if matrix is not None
                                 else cv2.resize(mask, (size, size), interpolation=cv2.INTER_NEAREST))
                half = (0.5 * rgb.astype(np.float32) + 0.5 * defect.astype(np.float32)).astype(np.uint8)
                sev.append(score_grid(self.pc_grid(half, category, config)[0], bank, config)[0])
                if len(examples) < 6:
                    examples.append((fit_display(defect, 320), f"synthetic {kind} (current lot)"))
        scale = fit_score_scale(loo, syn)
        pixel_scale = fit_pixel_scale(good_maps, synthetic_maps=syn_maps, synthetic_masks=syn_masks)
        good_n, syn_n, sev_n = (normalize_scores(np.asarray(v), scale) for v in (loo, syn, sev))
        policy = default_policy({"cal": good_n, "syn": syn_n, "sev": sev_n, "sev_levels": np.full(len(sev_n), 0.5)},
                                costs or CostModel())
        return {"category": category, "config": config.to_dict(), "bank": np.ascontiguousarray(bank.bank.astype(np.float16)),
                "score_scale": scale, "pixel_scale": pixel_scale, "threshold": float(policy["threshold"]),
                "band": float(policy["band"]), "lot_shift": shift, "n_good": int(n),
                "added_patches": int(len(order)), "coreset_ratio": coreset_ratio,
                "good_scores": good_n.round(2).tolist(), "synthetic_scores": syn_n.round(2).tolist(),
                "synthetic_examples": examples}

    # --------------------------------------------------------- few-shot
    def onboard_product(self, good_images: Sequence, name: str = "new_product", image_size: int | None = None,
                        synthetic_per_image: int = 2, seed: int = 0, costs: CostModel | None = None) -> dict:
        """Build a PatchCore model for a new product from a handful of GOOD images only.

        Good-score distribution comes from leave-one-image-out scoring; synthetic
        defects generated from the same images calibrate the threshold.
        """
        if len(good_images) < 2:
            raise ValueError("Upload at least 2 good images (8-30 recommended).")
        image_size = int(image_size or self.manifest["feature_extractor"]["image_size"])
        config = PatchCoreConfig(backbone=self.manifest["feature_extractor"]["backbone"], image_size=image_size)
        rgbs = [to_rgb_array(i) for i in good_images]
        grids = np.stack([self.patch_grid(rgb, image_size) for rgb in rgbs])
        n, h, w, c = grids.shape
        flat = grids.reshape(-1, c)
        owner = np.repeat(np.arange(n), h * w)
        loo_scores, good_maps = [], []
        for i in range(n):
            bank = MemoryBank(flat[owner != i])
            raw, amap, _ = score_grid(grids[i], bank, config)
            loo_scores.append(raw)
            good_maps.append(amap)
        ratio = 1.0 if len(flat) <= 20000 else 20000 / len(flat)
        bank_vectors = build_memory_bank(grids, coreset_ratio=ratio, seed=seed)
        bank = MemoryBank(bank_vectors)
        rng = np.random.default_rng(seed)
        syn_scores, syn_examples = [], []
        for i, rgb in enumerate(rgbs):
            for j in range(synthetic_per_image):
                defect, mask, kind = make_synthetic_defect(rgb, None, int(rng.integers(1 << 30)))
                raw, _, _ = score_grid(self.patch_grid(defect, image_size), bank, config)
                syn_scores.append(raw)
                if len(syn_examples) < 6:
                    syn_examples.append((fit_display(defect, 320), f"synthetic {kind}"))
        scale = fit_score_scale(loo_scores, syn_scores)
        pixel_scale = fit_pixel_scale(good_maps)
        good_norm = normalize_scores(loo_scores, scale)
        syn_norm = normalize_scores(syn_scores, scale)
        labels = np.r_[np.zeros(len(good_norm)), np.ones(len(syn_norm))]
        policy = select_threshold_by_cost(np.r_[good_norm, syn_norm], labels, costs or CostModel())
        return {"name": name, "config": config.__dict__, "bank": bank_vectors, "score_scale": scale,
                "pixel_scale": pixel_scale, "threshold": policy["threshold"], "band": policy["band"],
                "good_scores": good_norm.tolist(), "synthetic_scores": syn_norm.tolist(),
                "n_good": n, "bank_vectors": int(len(bank_vectors)), "synthetic_examples": syn_examples}

    def inspect_onboarded(self, image, model: dict, threshold: float | None = None,
                          band: float | None = None) -> dict:
        rgb = to_rgb_array(image)
        config = PatchCoreConfig(**model["config"])
        started = time.perf_counter()
        raw, anomaly_map, _ = score_grid(self.patch_grid(rgb, config.image_size, config.pool_size),
                                         MemoryBank(model["bank"]), config)
        score = float(normalize_scores(raw, model["score_scale"]))
        threshold = float(model["threshold"] if threshold is None else threshold)
        band = float(model["band"] if band is None else band)
        display = fit_display(rgb, 512)
        amap = np.clip(_resize_map(normalize_map(anomaly_map, model["pixel_scale"])), 0, 100)
        regions = find_regions(amap, out_size=(display.shape[1], display.shape[0]))
        decision = decide(score, threshold, band)
        heat = overlay(display, amap)
        return {"decision": decision, "score": score, "threshold": threshold, "band": band,
                "overlay": draw_regions(heat, regions) if decision != "PASS" else heat, "anomaly_map": amap,
                "regions": regions, "latency_ms": 1000 * (time.perf_counter() - started)}
