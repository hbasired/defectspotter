# 🔎 DefectSpotter — visual inspection without defect labels

**Project 3 · Manufacturing.** Factories rarely have many images of defects; most parts are good. DefectSpotter learns
**only from good parts**, flags anything unusual, **shows where** the anomaly is and turns it into a
**PASS / CHECK / REJECT** decision for the quality inspector at the end of the line — cutting both missed defects
(escapes) and false rejects.

* **Live app:** https://huggingface.co/spaces/bhkr0901/defectspotter (direct: https://bhkr0901-defectspotter.hf.space) —
  the same Gradio dashboard runs locally with `python app.py`.
* **Notebook:** [`DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb`](DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb) — EDA → training → evaluation → dashboard → deployment.
* **Data:** MVTec AD (bottle, screw, carpet; metal nut for few-shot onboarding) — CC BY-NC-SA 4.0, research/teaching only.
* **Complete guide:** [`docs/DefectSpotter_Complete_Guide.html`](docs/DefectSpotter_Complete_Guide.html) — one self-contained
  page (download it and open it in a browser, or use the app's 📖 Guide tab): flow diagrams of every process, every model,
  concept and parameter, the five detectors compared, results, app tour with screenshots and the demo script.

<!-- RESULTS:START -->
## Results

### Per-product PatchCore recipe (chosen without real defects)
| product | input | pose | descriptors | good parts in bank | bank | score | label-free objective |
|---|---|---|---|---|---|---|---|
| bottle | 224 px | as captured | raw | as captured | 1,309 vectors (1.0 MB) | max patch | 0.9361 |
| screw | 288 px | normalised | raw | as captured | 3,318 vectors (2.4 MB) | top 1% of map | 0.9492 |
| carpet | 224 px | as captured | raw | as captured | 1,756 vectors (1.3 MB) | top 5% of map | 0.7941 |

### Detection and localisation (MVTec test images)
| category | method | image AUROC [95% CI] | pixel AUROC | AUPRO@30% |
|---|---|---|---|---|
| bottle | PatchCore | 1.000 [1.000–1.000] | 0.980 | 0.945 |
| bottle | Feature VAE | 0.997 [0.988–1.000] | 0.974 | 0.905 |
| bottle | Pixel VAE | 0.902 [0.798–0.972] | 0.897 | 0.723 |
| bottle | Mean baseline | 0.975 [0.943–0.996] | 0.954 | 0.836 |
| bottle | Ensemble | 1.000 [1.000–1.000] | 0.980 | 0.945 |
| screw | PatchCore | 0.997 [0.992–1.000] | 0.996 | 0.972 |
| screw | Feature VAE | 0.798 [0.712–0.877] | 0.978 | 0.905 |
| screw | Pixel VAE | 0.477 [0.366–0.590] | 0.959 | 0.830 |
| screw | Mean baseline | 0.277 [0.192–0.363] | 0.636 | 0.240 |
| screw | Ensemble | 0.994 [0.987–0.999] | 0.996 | 0.974 |
| carpet | PatchCore | 0.940 [0.893–0.974] | 0.983 | 0.931 |
| carpet | Feature VAE | 0.902 [0.848–0.949] | 0.983 | 0.900 |
| carpet | Pixel VAE | 0.671 [0.558–0.779] | 0.914 | 0.697 |
| carpet | Mean baseline | 0.435 [0.337–0.537] | 0.744 | 0.593 |
| carpet | Ensemble | 0.938 [0.890–0.973] | 0.985 | 0.932 |

### Mean over the three products
| method | image_auroc | pixel_auroc | aupro_30 |
|---|---|---|---|
| Ensemble | 0.978 | 0.987 | 0.95 |
| Feature VAE | 0.899 | 0.978 | 0.903 |
| Mean baseline | 0.562 | 0.778 | 0.556 |
| PatchCore | 0.979 | 0.986 | 0.949 |
| Pixel VAE | 0.683 | 0.923 | 0.75 |

### New production lot: label-free lot-shift test and re-qualification (operating split)
| category | n_good_current_lot | lot-shift test | good parts → CHECK | escapes | false rejects |
|---|---|---|---|---|---|
| bottle | 7 | p = 0.385 → no change | — | — | — |
| screw | 14 | p = 0.862 → no change | — | — | — |
| carpet | 10 | p = 0.00569 → re-qualified | 50.0% → 5.6% | 5.3% → 8.8% | 0.0% → 0.0% |

### Operating point on the held-out operating split (1% prevalence cost model)
| category | method | policy | threshold ± band | false rejects | escapes | CHECK load | cost / 1,000 parts |
|---|---|---|---|---|---|---|---|
| bottle | PatchCore | real_oracle_cost | 41.5 ± 5.0 | 0.0% | 0.0% | 0.1% | 0 |
| bottle | PatchCore | synthetic_cost | 39.0 ± 5.0 | 0.0% | 0.0% | 0.0% | 0 |
| bottle | Feature VAE | real_oracle_cost | 31.5 ± 2.5 | 7.7% | 0.0% | 15.2% | 198 |
| bottle | Feature VAE | synthetic_cost | 33.0 ± 2.5 | 7.7% | 0.0% | 0.0% | 152 |
| bottle | Ensemble | real_oracle_cost | 41.5 ± 5.0 | 0.0% | 0.0% | 0.1% | 0 |
| bottle | Ensemble | synthetic_cost | 39.0 ± 5.0 | 0.0% | 0.0% | 0.0% | 0 |
| screw | PatchCore | real_oracle_cost | 33.0 ± 2.5 | 0.0% | 5.1% | 0.1% | 205 |
| screw | PatchCore | synthetic_cost | 36.5 ± 5.0 | 0.0% | 6.4% | 0.2% | 257 |
| screw | Feature VAE | real_oracle_cost | 28.0 ± 0.0 | 37.0% | 17.9% | 0.0% | 1451 |
| screw | Feature VAE | synthetic_cost | 46.5 ± 7.5 | 7.4% | 64.1% | 3.9% | 2722 |
| screw | Ensemble | real_oracle_cost | 32.5 ± 2.5 | 0.0% | 5.1% | 3.8% | 216 |
| screw | Ensemble | synthetic_cost | 33.0 ± 2.5 | 0.0% | 6.4% | 3.8% | 268 |
| carpet | PatchCore | real_oracle_cost | 37.0 ± 0.0 | 5.6% | 10.5% | 0.0% | 531 |
| carpet | PatchCore | synthetic_cost | 35.5 ± 5.0 | 0.0% | 5.3% | 49.6% | 359 |
| carpet | Feature VAE | real_oracle_cost | 30.5 ± 0.0 | 11.1% | 12.3% | 0.0% | 711 |
| carpet | Feature VAE | synthetic_cost | 30.5 ± 2.5 | 0.0% | 10.5% | 38.6% | 537 |
| carpet | Ensemble | real_oracle_cost | 35.0 ± 0.0 | 0.0% | 10.5% | 0.0% | 421 |
| carpet | Ensemble | synthetic_cost | 34.5 ± 5.0 | 0.0% | 5.3% | 55.2% | 376 |

### Threshold chosen with synthetic defects vs real defects (0–100 scale)
| category | method | synthetic_threshold | real_threshold | gap_synthetic_vs_real | diffusion_threshold | gap_diffusion_vs_real |
|---|---|---|---|---|---|---|
| bottle | PatchCore | 39.0 | 41.5 | -2.5 | 39.0 | -2.5 |
| bottle | Feature VAE | 33.0 | 31.5 | 1.5 | 33.0 | 1.5 |
| screw | PatchCore | 36.5 | 33.0 | 3.5 | 35.5 | 2.5 |
| screw | Feature VAE | 46.5 | 28.0 | 18.5 | 56.5 | 28.5 |
| carpet | PatchCore | 35.5 | 37.0 | -1.5 | 35.5 | -1.5 |
| carpet | Feature VAE | 30.5 | 30.5 | 0.0 | 30.5 | 0.0 |

### Latency per part (CPU, ONNX Runtime, end-to-end)
| category | method | latency_ms_median | latency_ms_p95 |
|---|---|---|---|
| bottle | PatchCore | 88.0 | 119.0 |
| bottle | Feature VAE | 68.0 | 98.0 |
| bottle | Pixel VAE | 35.0 | 37.0 |
| bottle | Mean baseline | 34.0 | 49.0 |
| bottle | Ensemble | 148.0 | 178.0 |
| screw | PatchCore | 136.0 | 172.0 |
| screw | Feature VAE | 56.0 | 69.0 |
| screw | Pixel VAE | 30.0 | 31.0 |
| screw | Mean baseline | 68.0 | 78.0 |
| screw | Ensemble | 166.0 | 223.0 |
| carpet | PatchCore | 84.0 | 115.0 |
| carpet | Feature VAE | 59.0 | 75.0 |
| carpet | Pixel VAE | 29.0 | 34.0 |
| carpet | Mean baseline | 26.0 | 28.0 |
| carpet | Ensemble | 155.0 | 199.0 |

### Few-shot onboarding (metal nut)
| shots | image_auroc | pixel_auroc |
|---|---|---|
| 1.0 | 0.731 | 0.914 |
| 2.0 | 0.822 | 0.923 |
| 4.0 | 0.906 | 0.949 |
| 8.0 | 0.979 | 0.973 |
| 16.0 | 0.987 | 0.977 |
| 32.0 | 0.987 | 0.98 |
| 64.0 | 0.989 | 0.981 |
| 176.0 | 0.991 | 0.982 |

Cost model: prevalence 1%, escape 400, false reject 2, CHECK 0.30; no inspection = 4000, 100% manual = 1100 per 1,000 parts.
<!-- RESULTS:END -->

**Reading the tables.** PatchCore is the deployed detector (the ensemble's label-free weight is 1.0, i.e. PatchCore
alone). The VAE fails the label-free capability check — it cannot separate even full-strength synthetic defects from
good parts (synthetic AUROC ≈ 0.6) — so its cost-optimal default degenerates to "reject everything"; it is kept as
the brief's GenAI comparison, not as a deployable detector. Carpet's 50% CHECK load is the MVTec lot change
(see *How the v2 recipe was chosen*): false rejects stay at 0% because uncertain parts go to a human.


## What the app does

The app is a **presentation that ends in a live demonstration**: every page opens with a "what you see / how to read
it" box, every chart has a caption, and the global settings (product, detector, production lot, threshold, CHECK band)
apply everywhere.

| Tab | What you can do |
|---|---|
| 🏠 **Overview** | Problem, user, how it works, headline results, deployed operating point per product, business case, responsible AI |
| 📊 **Data** | Where the data comes from (MVTec AD, pinned mirror, licence), the leakage-free split, EDA charts, synthetic & diffusion defects |
| 🧠 **Models & training** | What is trained on what (backbone frozen, no head), **the five detectors and how they differ**, backbone, PatchCore recipe per product, both VAEs with every parameter, training curves, baseline, calibration |
| 📈 **Results** | **Model comparison — which detector is best** (all five on the same held-out defects, verdict), metric glossary, **confusion-matrix heat maps** of the deployed decisions, AUROC bars/ROC/heat map, per-defect-type decisions, qualitative heat maps, thresholds, latency |
| 🎚️ **Threshold & cost** | The threshold trade-off priced in money; confusion matrix at your threshold; apply the cost-optimal threshold computed without real defects |
| 🌗 **Drift & new lots** | Lighting/focus/camera simulation with the camera-health gate, 200-part stream with EWMA + PSI alarms, drift heat maps, **lot-shift test and live re-qualification** |
| 🆕 **New product** | Few-shot onboarding of the metal nut from a handful of good images |
| 📖 **Guide** | The complete guide (`docs/GUIDE.md`) + download of the complete HTML guide with flow diagrams |
| 🎬 **Live demo** (last) | Inspect a part (upload / webcam / samples) with decision card, heat map, all detectors, VAE reconstruction · Batch · Review queue (human in the loop, audit log) · API |

Images to upload during a demonstration, with their expected outcomes: [`demo_images/`](demo_images/README.md)
(also as `demo_images.zip`).

## How it works

```mermaid
flowchart LR
  A[Good parts only] --> B[Frozen ResNet-18, not fine-tuned<br/>layer2+3 patch features<br/>ONNX ~11 MB, 2 outputs]
  B --> C[Per-product recipe<br/>size · 3x3/raw · rotations<br/>greedy coreset bank fp16]
  B --> F[Feature-space VAE<br/>spatial latent, ONNX]
  A --> V[Pixel VAE v1<br/>L1+SSIM, ONNX]
  A --> M[ECC-aligned<br/>mean good image]
  P[Held-out good parts] --> S[Synthetic defects<br/>cut-paste · scratch · stain<br/>+ SD inpainting]
  C --> K[kNN distance / reconstruction error → heat map → score]
  F --> K
  V --> K
  M --> K
  S --> T[0-100 scale + cost-optimal<br/>PASS/CHECK/REJECT thresholds]
  K --> D{Decision}
  T --> D
  D -->|CHECK| H[Human review + audit log]
  D --> R[Camera-health gate + EWMA/PSI drift alarms]
  R --> Q[Lot-shift test → re-qualification<br/>with good parts of the new lot]
```

1. **PatchCore-style detector (primary).** A frozen ImageNet ResNet-18 (`layer2` + `layer3`) turns every image into a
   grid of 384-d patch descriptors. One shared ONNX file (≈11 MB, dynamic resolution) outputs both the PatchCore **3×3
   neighbourhood average** and the **raw descriptors**. Each product gets its own recipe, chosen label-free: input
   resolution (224 / 288 px), 3×3 vs raw descriptors (raw keeps fine weave detail on carpet) and, for objects that arrive in
   any orientation, **90°-rotated good parts in the bank** (rotations keep a screw's thread handedness; flips would not).
   All good training patches form a memory bank compressed by **greedy k-center coreset selection** (keeps
   rare-but-normal patches that random sampling would drop). A test patch's distance to its nearest normal patch is its
   anomaly; the map is upsampled and smoothed, the image score is the max / top-k mean.
2. **Variational autoencoders (Generative AI).** The **feature-space VAE** (v3) is trained on good parts' CNN feature
   maps (384 × 28 × 28 → spatial latent 32 × 7 × 7 → back; feature MSE + β·KL) and its reconstruction error is the
   generative detector. The **pixel VAE** (v1: 128 px images, 128-d latent, L1 + SSIM + β·KL) is kept for comparison and
   for its reconstruction view — its blurry reconstructions show why working in feature space matters.
3. **Classical baseline.** ECC alignment (translation / rigid / affine chosen on synthetic data) to a median good reference,
   then a per-pixel z-score against the mean and standard deviation of aligned good parts.
4. **Synthetic defects choose the threshold.** Procedural **cut-paste / scratch / stain** defects and
   **Stable-Diffusion refinement** of them (SD-1.5 image-to-image SDEdit + LCM-LoRA, strength 0.45 on a crop around each
   defect, product-specific prompts, masked composite — plain inpainting just re-drew a good surface) on held-out
   good parts. Each synthetic defect is also replayed at **25% / 50% severity** (detection-limit calibration — real screw
   thread damage is far subtler than full-strength synthetic defects); a set joins the default calibration only if the
   detector separates it from good parts (synthetic AUROC ≥ 0.9, a label-free capability check). With held-out good parts they set the score scale
   (25 = typical good, 75 = typical synthetic defect), tune hyper-parameters label-free and pick the **cost-optimal
   three-way thresholds**. Real defects are used only to measure how far that threshold is from an oracle chosen with labels.
5. **Decisions with economics.** `PASS < t − b ≤ CHECK < t + b ≤ REJECT`; `(t, b)` minimise
   `(1−p)·FRR·C_false_reject + p·escape·C_escape + review·C_check` with inspector capacity ≤ 10% of parts. The rates are
   estimated on kernel-smoothed score distributions: with ~50 held-out good parts an empirical false-reject rate moves in
   2% steps and the optimum would snap to single samples.
6. **Responsible operation.** No automatic decision on images outside the qualified acquisition range (brightness /
   contrast / focus z-scores) — they go to CHECK; EWMA control charts and PSI on the score stream and camera-health
   signals raise drift alarms; every CHECK is decided and logged by a human.

## What changed compared with the first notebook (critical review)

| Issue in the original notebook | Fix |
|---|---|
| "Coreset" was **random** subsampling | real greedy k-center coreset (PatchCore) — unit-tested to keep rare clusters |
| MobileNetV3-Small backbone (screw AUROC 0.77, carpet 0.78 in our study) | **backbone study** → ResNet-18 (screw 0.93) within the 10–20 MB ONNX budget |
| Never executed, no results; 109 KB base64 zip inside the notebook | full run with real numbers; notebook clones the repo and runs readable stages |
| Space needed PyTorch (+ FAISS) at runtime | **ONNX-only runtime** (extractor + VAE) — light, fast installs |
| Score scale anchored on combined quantiles; saturated heat maps | label-free anchors (good / synthetic medians) and a piecewise pixel scale |
| HPO objective could not separate configs (synthetic AUROC saturates at 1.0) | adds TPR at ≤1% false rejects, the operating requirement |
| PSI-only drift demo; health signals not used for decisions | camera-health **gate** + EWMA control charts + PSI; maintenance-aware stream |
| Diffusion defects disabled by default and never compared | SD-1.5 SDEdit refinement of procedural defects (CPU-feasible with LCM-LoRA), capability-checked and compared with real defects |
| Thresholds set by conspicuous synthetic defects only (first run: 35% screw escapes) | detection-limit calibration: synthetic defects replayed at 25/50/100% severity; v1 kept for comparison |
| No human review workflow | review queue + audit log; ground truth shown on samples |
| v1: one fixed recipe (224 px, 3×3 averaging) for every product — carpet AUROC 0.88, screw 0.93 | **v2:** per-product label-free search over resolution, 3×3 vs raw descriptors and rotation-augmented banks; subtle (50%) synthetic defects join the selection objective; parsimony rule → **screw 0.983, carpet 0.940**, banks 1–2.4 MB |
| v1: default thresholds from raw counts on ~50 good parts; reports from mixed runs (FRR up to 100% on carpet) | **v2:** smoothed cost optimum, CHECK band planned at ≤ 50% of inspector capacity, one shared recipe for pipeline and app, single consistent run; label-free vs real-defect threshold gap 9.7 → ≤ 2.5 points; ensemble weight chosen on subtle defects (v1 ensemble was below PatchCore on screw) |
| v1: synthetic object defects mostly on the background (mask forced ≥ 45% "foreground"; a screw is ~13% of the frame) | Otsu-based part mask: defects land on the part; diffusion-refined defects are now detectable (AUROC 0.86–1.0 vs 0.4–0.6) and join the calibration |
| v2: screws at random orientation (memory must cover every angle; 9% escapes) | **v3: pose normalisation** for elongated parts (principal axis horizontal, head left, part extent centred; maps warped back exactly) → screw AUROC **0.997**, escapes 6.4% |
| v1/v2: pixel VAE barely beats chance on screw (0.48) | **v3: feature-space VAE** (reconstructs the backbone's feature map, spatial latent) → mean AUROC 0.68 → **0.90** |
| v2: carpet's new lot sent ~50% of good parts to CHECK | **v3: label-free lot-shift test + re-qualification** from 10 good parts of the new lot → CHECK **5.7%**, false rejects 0% |
| Dual-axis charts, dark hero | single-axis charts, colour-blind-validated palette, light theme |
| ZeroGPU claim without `@spaces.GPU` | ZeroGPU-compatible entry point; deployment script with clear free-tier rules |

## How the v2 recipe was chosen (and what the test set was used for)

Transparency matters for a detector that is supposed to work without defect labels, so here is the full path:

1. **Exploratory study (like the backbone study).** Test-set AUROC was used *only* to propose candidate design options:
   raw descriptors for textures, 288 px input, 90°-rotated good parts in the bank, pose normalisation for elongated parts,
   a feature-space VAE. This is a mild optimistic bias on the reported test numbers, the same kind the backbone study
   carries.
2. **Label-free selection.** Within that search space every product's recipe is selected on held-out good parts +
   synthetic defects (full-strength and 50%-strength), never on real defects, with a parsimony rule: among recipes within a
   tolerance of the best objective, deploy the one with the cheapest nearest-neighbour search.
3. **One revision after the first full run (disclosed).** The tolerance was first 0.001. Carpet's 288 px recipe then won by
   0.003 on synthetic data but turned out far more sensitive to the *lot change* between MVTec's carpet train and test
   captures (test images are ~4 grey levels brighter and come from a different roll; test-good false rejects 39% vs 0% at
   224 px). Differences of 0.003 are well inside the sampling noise of these metrics (standard error ≈ 0.03 on ~50 good and
   ~250 synthetic parts), so the tolerance was raised to 0.005; this changed only carpet's choice. Synthetic jitter
   (±5% brightness, slight defocus) did **not** reproduce the lot change, i.e. no label-free check we tried could have
   caught it — a real line catches it with the CHECK-rate / score drift alarms and re-qualifies on good parts from the new lot.
4. **Lot re-qualification** is triggered only by the label-free lot-shift test (good parts only, p < 0.01): it fires for
   carpet and not for bottle or screw — even though re-qualifying screw would also have lowered its escapes on the test
   split, the rule was fixed in advance and is not picked from test results.
5. **Default thresholds** come from the smoothed cost optimum on held-out good + capability-gated synthetic sets, with
   the CHECK band planned at ≤ 50% of inspector capacity. Real defects only enter the separate `real_oracle_cost` policy,
   chosen on the oracle split and evaluated, like every policy, on the disjoint operating split.

## Real-world numbers

The Overview turns the deployed decisions into **one 8-hour shift** (2,500 parts/hour, 1% defective; human check 36 s at
30 USD/h ≈ BLS median inspector pay + overhead; inspectors catch ~80% of what they are shown; escape 400 USD per the
1-10-100 cost-of-quality rule — all editable in *Threshold & cost*). Carpet before re-qualification would need **12.4
inspectors per shift**; after, **1.4** (bottle 0.0, screw 0.1), against **25 inspectors** for 100% manual inspection, which
still misses ~40 defects per shift. Details and sources: [`docs/GUIDE.md`](docs/GUIDE.md#10c-real-world-numbers-one-production-shift).

## Which detector is best?

All five detectors in the app's *Detector* menu were built from good parts only and calibrated the same label-free way:

| detector | family | mean image AUROC | pixel AUROC | cost / 1,000 parts | verdict |
|---|---|---|---|---|---|
| **PatchCore** | memory bank of normal patches | **0.979** | 0.986 | **206 USD** | 🏆 best — deployed |
| Ensemble (PatchCore + feature VAE) | combination | 0.978 | 0.987 | 215 USD | tie, but ~1.8× slower — not deployed |
| Feature-space VAE | generative AI on backbone features | 0.899 | 0.978 | 1,137 USD | best generative model |
| Pixel VAE (v1) | generative AI on pixels | 0.683 | 0.923 | 1,855 USD | superseded by the feature VAE |
| Mean-image baseline | classical statistics | 0.562 | 0.778 | 1,930 USD | classical reference |

Cost: expected cost per 1,000 parts at 1% defects at each detector's label-free threshold (qualification-lot models).

**Best detector per product** (“=” is a tie within 0.0005; costs at each detector's own label-free threshold):

| product | best ranking (image AUROC) | best heat map (pixel AUROC · AUPRO) | lowest cost / 1,000 parts | better VAE | use |
|---|---|---|---|---|---|
| bottle | PatchCore = Ensemble (1.000) | PatchCore = Ensemble (0.980 · 0.945) | PatchCore = Ensemble (0.15 USD) | feature-space (0.997 vs 0.902) | **PatchCore** (the ensemble *is* PatchCore here, w = 1.0) |
| screw | PatchCore (0.997) | PatchCore = Ensemble (0.996) · Ensemble (0.974) | PatchCore (257 USD) | feature-space (0.798 vs 0.477) | **PatchCore** — pose normalisation; the VAEs fail on rotation |
| carpet | PatchCore (0.940) | Ensemble (0.985 · 0.932) | PatchCore (359 USD) | feature-space (0.902 vs 0.671) | **PatchCore, re-qualified lot** (CHECK 49.6% → 5.7%) |

**Overall: PatchCore** — best ranking and lowest cost on every product; the ensemble's heat-map edge on screw and carpet
(≤ 0.002) is a statistical tie that does not pay for running two models; the feature-space VAE beats the pixel VAE on
every product.
The app shows the full table (per product, false rejects, escapes, CHECK share, speed) in 🧠 Models & training and
📈 Results.

## Guide

[`docs/DefectSpotter_Complete_Guide.html`](docs/DefectSpotter_Complete_Guide.html) is the complete, self-contained guide
(27 chapters, flow diagrams, screenshots; rebuilt from the reports by `python scripts/build_guide.py`).
[`docs/GUIDE.md`](docs/GUIDE.md) explains everything step by step: the problem, the data and split, the frozen
ResNet-18 backbone (not fine-tuned, no head), PatchCore and the coreset, pose normalisation, both VAEs (concepts, every
layer and parameter), synthetic and diffusion defects, calibration and the cost model, metrics, drift monitoring, lot
re-qualification, few-shot onboarding, the demo script and a glossary. The same guide is the app's 📖 Guide tab.

## Run it locally

```bash
git clone https://github.com/hbasired/defectspotter.git && cd defectspotter
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt                         # runtime only - no PyTorch needed
python app.py                                           # → http://127.0.0.1:7860   (--share for a public link)
```

Trained models (`artifacts/`, ≈80 MB) and the evidence the dashboard plots (`reports/`, `examples/`) are in the repository,
so the app works immediately after cloning. Python 3.10–3.13.

### Reproduce the whole experiment

```bash
pip install -r requirements-train.txt
python scripts/download_data.py --dest data/mvtec_ad                 # pinned HF mirror of MVTec AD (1.3 GB)
python scripts/train.py --data data/mvtec_ad --vae-epochs 120        # ~2 h on a 4-core CPU, ~25 min on a GPU
pip install -r requirements-diffusion.txt                            # optional GenAI defects
python scripts/generate_diffusion_defects.py --data data/mvtec_ad
python scripts/train.py --data data/mvtec_ad --stages severity diffusion calibrate evaluate drift summary
python scripts/backbone_study.py --data data/mvtec_ad                # optional, ~25 min on CPU
pip install -r requirements-dev.txt && pytest                        # unit + integration tests (also run by CI)
```

### Deploy (Hugging Face, free ZeroGPU tier)

```bash
export HF_TOKEN=hf_xxx                       # write token - never commit it
python scripts/deploy_hf.py --space bhkr0901/defectspotter
```

Free personal accounts (verified e-mail, account older than 30 days) can host up to two Gradio Spaces on ZeroGPU; CPU-basic
Gradio Spaces now need PRO. All inference runs on CPU with ONNX Runtime, so visitors never use GPU quota. Details:
[`DefectSpotter_HuggingFace_Deployment_Guide.md`](DefectSpotter_HuggingFace_Deployment_Guide.md).

### API

```python
from gradio_client import Client, handle_file
client = Client("http://127.0.0.1:7860")   # or "bhkr0901/defectspotter"
record, overlay = client.predict(handle_file("part.png"), "bottle", "patchcore", None, None, api_name="/inspect_part")
print(record["decision"], record["score"], record["regions"])
```

## Repository layout

```
app.py                      entry point (local, Colab, Hugging Face ZeroGPU)
defectspotter/              library: data, synthetic, features*, patchcore, vae*, baseline, metrics, decision,
                            drift, runtime (ONNX-only inference), pipeline*   (* = needs PyTorch, training only)
ui/                         Gradio app: theme, charts (Plotly), story (presentation), store (reports), app
scripts/                    download_data, train, generate_diffusion_defects, backbone_study, deploy_hf, summarize_results
artifacts/                  ONNX feature extractor + per-product memory bank, VAE (ONNX), baseline, calibration.json
reports/                    every metric/table/figure the app and README show (CSV/JSON/PNG)
examples/                   down-scaled MVTec test images (CC BY-NC-SA 4.0) for the demo
docs/                       model card, data card, screenshots
tests/                      unit + integration tests (GitHub Actions: .github/workflows/tests.yml)
```

## Responsible AI and limitations

* CHECK is a first-class outcome routed to a trained inspector; the audit log stores verdict, note, score, threshold and detector.
* The score is an anomaly **ranking**, not a defect probability; heat maps show visual deviation, not root cause.
* Lighting, focus and camera-position changes make good parts look anomalous — monitored with the camera-health gate,
  EWMA charts and PSI; alarms trigger maintenance and re-qualification, never silent retraining. Geometric camera shifts
  are **not** caught by brightness/sharpness signals (see the Drift tab) — the score chart is needed for those.
* Synthetic defects calibrate thresholds but do not prove coverage of every real failure mode.
* **Lot changes need re-qualification.** MVTec's carpet test images come from a brighter, different roll: the detector
  still ranks parts well, but thresholds learned on the training lot send many good test carpets to CHECK. The system
  fails safe (human review instead of scrapping) and the CHECK-rate alarm fires; the fix is re-qualifying on good parts
  from the new lot (and regenerating synthetic defects from them), not silent retraining.
* MVTec AD is non-commercial (CC BY-NC-SA 4.0). A production line needs plant-owned data, a qualified camera, lot/time-separated
  validation of every critical defect mode, access control and a documented response to CHECK, drift and camera failure.

See [`docs/MODEL_CARD.md`](docs/MODEL_CARD.md) and [`docs/DATA_CARD.md`](docs/DATA_CARD.md).

## Citation

Bergmann, Batzner, Fauser, Sattlegger, Steger. *The MVTec Anomaly Detection Dataset.* IJCV 2021 · Roth et al.
*Towards Total Recall in Industrial Anomaly Detection (PatchCore).* CVPR 2022 · Li et al. *CutPaste.* CVPR 2021 ·
Kingma & Welling. *Auto-Encoding Variational Bayes.* ICLR 2014 · Luo et al. *LCM-LoRA.* 2023.

Code: MIT (see `LICENSE`). Data and derived models: CC BY-NC-SA 4.0.
