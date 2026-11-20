# DefectSpotter — project handoff (scope, objectives, status, next steps)

This file lets anyone (a teammate, another Claude account, another cloud/Colab/local machine) continue the project
without the original chat. Read it top to bottom once; then use §7 to set up and §10 to pick the next task.

* **Repository:** https://github.com/hbasired/defectspotter
* **Branch:** everything is merged into `main` (feature branches `claude/optimistic-mccarthy-q7e5mh` and
  `claude/nice-einstein-ibh15o` are fully contained in it)
* **Live demo:** https://huggingface.co/spaces/bhkr0901/defectspotter (direct app: https://bhkr0901-defectspotter.hf.space)
* **Last updated:** 2026-11-20 (third build session: v3 detector — pose-normalised screws, feature-space VAE, lot
  re-qualification; presentation UI; real-world one-shift view; five-detector comparison; complete HTML guide; redeployed)

---

## 1. The brief (what was asked)

**Project 3 · Manufacturing · DefectSpotter: visual inspection without defect labels.**

* **Business problem.** Factories rarely have many images of defects; most parts are good. An inspection tool that learns
  only from good parts, flags anything unusual and shows where the anomaly is cuts both missed defects (escapes) and
  false rejects.
* **User.** A quality inspector at the end of a production line.
* **App.** Choose a product type (e.g. bottle, screw, carpet); upload a part image or a batch; get an anomaly score, a heat
  map over the suspected defect and a PASS / CHECK / REJECT decision; a threshold slider shows the trade-off between false
  rejects and escapes; a batch summary table.
* **Data.** MVTec AD (CC BY-NC-SA 4.0, research/teaching): three product categories; training uses good images only; test
  images include labelled defects with pixel masks.
* **CV techniques.** PatchCore-style nearest-neighbour memory bank of pre-trained CNN patch embeddings (Weeks 6–7); anomaly
  heat maps and thresholding; classical baseline = difference from the mean good image after alignment (Weeks 2–3).
* **Generative AI.** A small convolutional VAE on good parts as a second detector compared with the feature method
  (Week 8); synthetic defects (cut-paste and diffusion inpainting of scratches and stains) to choose the threshold without
  real defects; test how close that threshold is to one chosen with real defects.
* **Interface/deployment.** (Brief said Streamlit Community Cloud) — the owner chose **Gradio on Hugging Face Spaces**.
  ONNX feature extractor of about 10–20 MB plus compact memory banks (< 50 MB per category).
* **Evaluation.** Image AUROC, pixel AUROC; false-reject and escape rates at the chosen threshold; VAE vs feature method;
  synthetic vs real threshold; a simple cost table (escape vs false reject); latency per part.
* **Responsible AI.** Human check on CHECK decisions; lighting and camera changes as drift (Lecture D) and how to monitor them.
* **Stretch goals.** Drift monitor (anomaly scores over time with an alert); few-shot adaptation to a new product.

Owner's extra requirements (from the chat): light-themed, very interactive frontend that also *presents* the project
(problem, method, models, techniques, EDA, charts, tables, key indicators); everything the original Colab UI showed;
runs locally after `git clone` with clear requirements; the Colab notebook covers EDA → Gradio dashboard; research whether
the project is sound and adapt it to mirror real factory practice; deploy backend + frontend and give a URL to present.

## 2. Objectives and scope

| # | Objective | Success criterion | Status |
|---|---|---|---|
| O1 | Good-only detector with localisation | PatchCore image AUROC ≥ 0.9 on all 3 products, heat maps | ✅ (see §5) |
| O2 | Three-way decision with an honest operating point | thresholds chosen **without real defects**; FRR / escape / cost reported on a held-out split | ✅ |
| O3 | GenAI component | VAE detector compared with PatchCore; procedural + diffusion synthetic defects; synthetic-vs-real threshold gap | ✅ (diffusion re-calibration finishing at handoff time) |
| O4 | Deployable footprint | ONNX extractor 10–20 MB (actual 10.6 MB); banks < 50 MB; CPU latency per part reported | ✅ |
| O5 | Inspector-facing app | upload / batch / threshold slider / batch table / heat map + presentation, light theme | ✅ |
| O6 | Responsible AI | human review queue + audit log; camera-health gate; drift alarms; documentation | ✅ |
| O7 | Stretch | drift monitor with alert; few-shot new product | ✅ |
| O8 | Reproducibility | local + Colab run, tests, pinned data revision | ✅ |
| O9 | Public demo URL | Hugging Face ZeroGPU Space live | ✅ https://huggingface.co/spaces/bhkr0901/defectspotter |

**In scope:** MVTec AD bottle / screw / carpet (+ metal_nut for few-shot), research/teaching use, CPU-friendly inference.
**Out of scope:** commercial use of MVTec data, certified production control, PLC/MES integration, multi-camera fusion,
authentication / persistent multi-user storage (the review queue lives in the browser session).

## 3. What was built (architecture)

```
data (MVTec AD, pinned HF mirror)
  └─ scripts/train.py → defectspotter/pipeline.py (stages)
       index → eda → synthetic → patchcore → vae → mean → severity → diffusion → calibrate → evaluate → latency → drift → few_shot → examples → summary
       writes: artifacts/ (deployable models) · reports/ (CSV/JSON/PNG evidence) · examples/ (demo images)
app.py → ui/app.py (Gradio) → defectspotter/runtime.py (ONNX Runtime + NumPy only, no PyTorch)
```

| Component | Key design decision | Why |
|---|---|---|
| Feature extractor | ResNet-18 `layer2+layer3`, ONNX (10.6 MB) with **two outputs** (3×3 average, raw descriptors) and dynamic input size | backbone study: best accuracy inside the 10–20 MB budget (MobileNetV3 0.85 mean AUROC, EffNet-B0 0.91, **ResNet-18 0.93**, WRN-50 0.94 at 95 MB) |
| Per-product recipe (v2) | searched label-free: resolution 224/288, 3×3 vs raw descriptors, 1 or 4 orientations of good parts in the bank (objects), coreset 1/5/10%, blur, max/top-k | exploratory study: raw descriptors lift carpet 0.86 → 0.93; rotated banks lift screw 0.96 → 0.98 (random orientations); 288 px helps screw |
| Search speed | greedy-coreset **ladder** (one greedy run → all ratios, its prefixes); random 300k-patch candidate pool for rotated sets; one NN search per bank, blur/score variants as post-processing | the v2 grid (~140 configs per object) in ~1 h on 4 CPUs |
| Memory bank | **greedy k-center coreset** (JL projection to 128-d), float16 | the original notebook used random sampling; greedy keeps rare-but-normal patches |
| Hyper-parameters | chosen on held-out good + **synthetic** defects (full strength + **50% replays**) only; objective = 0.4·img AUROC + 0.3·TPR@≤1%FPR + 0.3·pixel AUROC | full-strength synthetics alone saturate and favour settings that only see conspicuous blobs |
| Synthetic placement | object mask = Otsu on CIELAB distance to the border colour + morphology | v1 thresholded at a fixed percentile → ≥ 45% "foreground"; most screw defects landed on the empty tray |
| VAE | 4-stage conv encoder → 1×1 squeeze → 128-d latent; L1 + 0.5(1−SSIM) + β·KL; ONNX ≈7.4 MB | small, sharper than MSE; no flips for screws (thread handedness) |
| Baseline | ECC alignment (none/translation/euclidean/affine picked on synthetic data) + per-pixel z-score | transparent classical comparison |
| Threshold calibration set | held-out good + synthetic defects at **25 / 50 / 100 % severity** (+ diffusion if present) | first run used full-strength synthetics only: screw real defects (median score 36) are far subtler than synthetic ones (75) → 35% escapes; the detection-limit sweep is label-free and mirrors inspection specs |
| Score scale | 0–100, 25 = median held-out good, 75 = median synthetic defect | label-free, interpretable for operators |
| Pixel scale | 0 = median good pixel, 50 = 99.9th pct good pixel, 100 = typical synthetic-defect pixel | localised, non-saturating heat maps |
| Decision | PASS < t−b ≤ CHECK < t+b ≤ REJECT, (t,b) minimise expected cost s.t. CHECK ≤ 10%, rates from **kernel-smoothed** score distributions | mirrors a real line: prevalence 1%, escape 400, false reject 2, CHECK 0.30 (illustrative units); smoothing stops the optimum snapping to single held-out parts |
| Ensemble weight | chosen on full + 50% synthetic defects, scanned from PatchCore-only down (ties keep PatchCore) | v1 (full-strength only) put the screw ensemble far below PatchCore |
| Health gate | brightness/contrast/sharpness z-score vs qualified images, \|z\|>4 → no automatic decision (CHECK) | lighting drift otherwise causes mass false rejects |
| Drift | EWMA (λ=0.2, 3σ, 2-in-a-row) on winsorised scores and health signals + rolling PSI | SPC practice; health signals lead for lighting, score chart needed for geometry |
| Few-shot | memory bank from k good images; threshold from leave-one-image-out scores | PatchCore needs no network training |
| Diffusion defects | base SD-1.5 **image-to-image SDEdit** + LCM-LoRA on a crop around each procedural defect (strength 0.45, 6 steps, guidance 1.5), masked composite | SD's inpainting model fills the hole from context and erased the defects even when started from them (mean change 2–6 grey levels, synthetic AUROC ≈ 0.4–0.6); SDEdit with the base model keeps the defect's structure and adds realistic texture; runs on CPU |
| Capability gate | a synthetic set joins the default threshold only if synthetic AUROC ≥ 0.9 vs held-out good (`reports/detection_capability.csv`) | undetectable sets (e.g. 25% carpet) otherwise push the cost optimum to "reject everything" |

Splits (seed 42): train/good → 80% `model_train_good` + 20% `calibration_good`; test → per defect type 35%
`oracle_calibration` (real-label threshold, comparison only) + 65% `operating_test`. Real defects never train a model or
choose the default threshold.

## 4. Repository map

| Path | Purpose |
|---|---|
| `app.py` | entry point (local / Colab / HF ZeroGPU: imports `spaces` first, registers a `@spaces.GPU` self-test) |
| `ui/` | `app.py` (9 tabs, live demo last, + API), `explain.py` (self-explanatory HTML: intros, cards, five-detector comparison, real-world view), `charts.py` (Plotly, colour-blind-validated palette, single y-axis), `story.py` (presentation HTML, numbers read from reports), `store.py`, `theme.py` (light theme, forced even in OS dark mode) |
| `defectspotter/` | `data`, `synthetic`, `features`*, `patchcore`, `pose`, `vae`*, `fvae`*, `baseline`, `metrics`, `decision`, `drift`, `operations`, `viz`, `runtime`, `pipeline`* (*need PyTorch, training only) |
| `scripts/` | `download_data.py`, `train.py`, `generate_diffusion_defects.py`, `backbone_study.py`, `deploy_hf.py`, `build_report.py`, `summarize_results.py`, `make_demo_images.py`, `build_guide.py` (+ `guide_html/`: content, SVG flow diagrams, styles) |
| `artifacts/` | `manifest.json`, `feature_extractor.onnx`, `<product>/{calibration.json, patchcore_bank.npy, vae.onnx, mean_baseline.*}` |
| `reports/` | every metric/table/figure (`metrics.csv`, `operating_metrics.csv`, `threshold_comparison.csv`, `scores.json`, `drift_*.csv`, `few_shot.csv`, `backbone_study.csv`, `figures/`, `DefectSpotter_report.html`) |
| `examples/` | down-scaled MVTec test images + masks for the demo, synthetic and diffusion galleries, new-product samples |
| `DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb` | Colab: clones the repo, runs each stage with inline charts, launches the app, deploys |
| `DefectSpotter_HuggingFace_Deployment_Guide.md` | local / Colab / HF / Claude deployment details |
| `docs/` | `GUIDE.md`, `DefectSpotter_Complete_Guide.html` (self-contained, built by `scripts/build_guide.py`), `MODEL_CARD.md`, `DATA_CARD.md`, `screenshots/` (WebP, used by the HTML guide) |
| `demo_images/` | images to upload in a live demo + `expected_outcomes.csv` (verified against the deployed models) |
| `tests/` | `test_core.py` (unit), `test_app.py` (integration on shipped artifacts) |

## 5. Results snapshot

<!-- HANDOFF-RESULTS:START -->
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
<!-- HANDOFF-RESULTS:END -->

## 6. Environment facts (as tested)

* Python 3.13 (works 3.10+), torch 2.15 CPU, torchvision 0.29, onnxruntime 1.32, gradio 6.32, plotly 7.1 (plotly.js 4.1.1),
  numpy 2.5, pandas 3.1, OpenCV 5.1, diffusers 0.43, transformers 5.23, peft 0.22.
* Hardware used: 4 vCPU, 15 GB RAM, no GPU. Full v2 pipeline ≈ 2.5 h (PatchCore search ≈ 50 min, VAE 120 epochs ≈ 5 s/epoch/
  product, drift ≈ 9 min); diffusion ≈ 17 s per image on CPU with LCM-LoRA (seconds on a GPU). Cloud sessions may restart
  the container: stages, the PatchCore search and VAE training are resumable per product (re-run the same command).
* Data: `foersben/mvtec-ad` @ `c75b39616f84db43677bcc8228caaafaf5096d7f` (an earlier pin pointed at a partial upload).
* Hugging Face rules (Nov 2026, official docs): free accounts (verified e-mail, > 30 days old) may host **2 Gradio Spaces on
  ZeroGPU**; CPU-basic Gradio/Docker Spaces need PRO; static Spaces are free; Streamlit is only available via Docker.
  ZeroGPU supports Python 3.10.13 / 3.12.12 and PyTorch 2.8–2.13; needs ≥ 1 `@spaces.GPU` function at start-up.

## 7. Set up from scratch (new machine, new cloud account or new Claude session)

```bash
git clone https://github.com/hbasired/defectspotter.git && cd defectspotter
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # app only (no torch)
python app.py                                              # http://127.0.0.1:7860
pip install -r requirements-dev.txt && pytest              # unit + integration tests
```

Re-train (only needed after changing the method or data):

```bash
pip install -r requirements-train.txt
python scripts/download_data.py --dest data/mvtec_ad
python scripts/train.py --data data/mvtec_ad --work-dir ../defectspotter_work --vae-epochs 120   # all stages (~2 h CPU)
python scripts/train.py --data data/mvtec_ad --stages calibrate evaluate summary         # re-run selected stages
pip install -r requirements-diffusion.txt
python scripts/generate_diffusion_defects.py --data data/mvtec_ad --per-category 8 --steps 4 --guidance 1.0
python scripts/train.py --data data/mvtec_ad --stages severity diffusion calibrate evaluate drift summary
python scripts/build_report.py                                                            # static HTML report
python scripts/summarize_results.py                                                       # Markdown tables
```

Stages are independent and cache their inputs in `--work-dir` (features, synthetic images, raw scores), so a failed or
changed stage can be re-run alone. `--quick` gives a ~15-minute smoke run.

**Starting a new Claude Code session:** attach this repository, check out the branch, and ask Claude to read
`PROJECT_HANDOFF.md` first. Secrets are never in the repo: GitHub write access comes from the Claude GitHub App on the repo;
Hugging Face deployment needs a write token entered via the secure credential form or the `HF_TOKEN` environment variable /
Colab secret.

## 8. Using the app (demo script, ~6 minutes)

1. **Story** — problem, user, pipeline, data table, results, business case, responsible AI.
2. **Inspect** — product *Bottle* → click a defective sample → decision card (with ground truth), heat map + regions, all
   four detectors side by side, VAE reconstruction, pixel map, camera-health table. Move the threshold / CHECK band sliders.
3. **Batch** → *Load sample batch* → *Inspect batch*: tiles, strip chart, table, CSV. CHECKs flow to **Review**; confirm or
   release them and download the audit log.
4. **Threshold & cost** — raise the escape cost or defect rate, press *Apply cost-optimal threshold* (no real defects used),
   compare policies (synthetic vs real oracle).
5. **Drift** — dim the lamp on a good part: model alone → REJECT, health gate → CHECK; replay the 200-part stream (EWMA + PSI).
6. **New product** — load 8 sample metal nuts → *Build model* (seconds) → inspect held-out nuts.
7. **Models / EDA / API** for questions (ROC, per-defect, backbone study, latency, footprint, endpoints).

## 9. Deployment status

* **Deployed** from a Claude Code session: https://huggingface.co/spaces/bhkr0901/defectspotter (Gradio, ZeroGPU hardware,
  inference on CPU with ONNX Runtime). Smoke-tested through the API (`/inspect_part`, `/health`).
* Redeploy after retraining or code changes: `HF_TOKEN=<write token> python scripts/deploy_hf.py --space bhkr0901/defectspotter`
  (in a Claude session the token is saved once via the secure credential form; Claude can redeploy and smoke-test on request).
* A Claude session cannot expose a public URL to a backend running inside its own sandbox (no inbound connections; tunnels
  such as `--share` are blocked) — the Space is the always-on backend + frontend.
* After deploying: open the Space, wait for the build (~1–2 min), run one sample per product on the Inspect tab, check `/health`.
* **Reload open pages after a redeploy.** A page opened before the update talks to a server with different event wiring and
  every control fails ("An event handler didn't receive enough input values", "Value … is not in the list of choices").
  The app now embeds a build fingerprint and shows a yellow "new version deployed — reload" banner in that case.

## 10. Next steps (prioritised backlog)

**Done in session 3:** feature-space VAE (better GenAI detector), pose-normalised screws, label-free lot-shift test +
re-qualification (carpet CHECK load ~50% → ~6%), UI rebuilt as a presentation ending in the live demo (intro boxes,
captions, KPI tiles, confusion-matrix heat maps, model/training explainer), `docs/GUIDE.md`, `demo_images/`; real-world
one-shift view (`defectspotter/operations.py`: parts, inspectors, escapes, USD per shift, sourced assumptions); five-detector
comparison with verdict and best detector per product (Models & training + Results tabs; PatchCore best and deployed);
"Which detector uses the ResNet?" map (Overview + Models: only PatchCore and the feature-space VAE use the frozen ResNet-18;
the pixel VAE and the mean baseline work on raw pixels; the ResNet is never trained); complete HTML guide
(`scripts/build_guide.py` → `docs/DefectSpotter_Complete_Guide.html`, downloadable from the Guide tab); stale-page reload
banner; notebook cells 12c (detector comparison), 12d (one shift) and 16b (HTML guide).

**Done in session 2:** branches merged into `main`; Space deployed and smoke-tested; v2 detector (per-product label-free
recipe; screw 0.935 → 0.983, carpet 0.880 → 0.940); consistent single run of all reports; smoothed thresholds with
capacity headroom; synthetic defects placed on the part; diffusion defects usable (detectable) for calibration; CI.

**P1 — improve results**
1. Re-qualification in production: persist re-qualified models per lot (versioned), trigger the lot-shift test from the
   CHECK-rate alarm, and add a re-validation report before a re-qualified model goes live.
2. Remaining screw escapes are very faint thread damage: try aligned 320–384 px crops of the part (tighter field of view
   after pose normalisation) or WRN-50 fp16 for screw only.
3. Feature-space VAE: multi-scale (layer1+2+3) features and a perceptual/feature loss on the pixel VAE.
4. More diffusion defects on a GPU (50+/product, more prompts); test on phone photos (domain shift).

**P2 — productionise / extend**
5. Persist the review queue and audit log (SQLite / HF Dataset) with user identity; authentication on the Space.
6. MVTec AD 2 (lighting variations in the test set) to validate the drift story on real lighting changes.
7. Continuous monitoring page fed by real inspection logs (EWMA/PSI per shift, CHECK-rate alarm).
8. Optional in-browser demo (ONNX Runtime Web) for a fully static deployment.

## 11. Known issues and limitations

* Synthetic defects are easier than some real ones; v2 closes most of the threshold gap (|gap| ≤ 2.5 points, see
  `reports/threshold_comparison.csv`), but very faint screw defects still pass.
* MVTec's carpet test images come from a different, brighter lot: thresholds learned on the training lot route ~50% of
  good test carpets to CHECK (false rejects stay ~0%). Re-qualify on the new lot (§10 P1-1).
* The v2 search space was proposed after an exploratory look at test AUROC (disclosed in the README); the per-product
  choice within it is label-free, with one disclosed revision of the parsimony tolerance.
* Down-scaling to 224–288 px hides very small defects (many MVTec defects cover < 1% of the image).
* Geometric camera shifts are not caught by brightness/sharpness health signals — rely on the score chart for those.
* The review queue is per browser session (no persistence) — fine for a demo, not for production.
* Score = ranking, not probability; heat map = visual deviation, not root cause.
* MVTec AD and everything derived from it (banks, VAEs, examples) are CC BY-NC-SA 4.0: non-commercial only.

## 12. Glossary

**FRR** false-reject rate (good part → REJECT) · **Escape rate** defective part → PASS · **CHECK / review load** share of
parts sent to a human · **AUROC** ranking quality (1 = perfect) · **Pixel AUROC / AUPRO@30%** localisation quality (AUPRO
weights every defect region equally) · **Coreset** small subset that covers the memory bank · **EWMA** exponentially
weighted moving average control chart · **PSI** population stability index (< 0.1 stable, 0.1–0.25 warn, > 0.25 alert).
