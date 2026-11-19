# DefectSpotter — project handoff (scope, objectives, status, next steps)

This file lets anyone (a teammate, another Claude account, another cloud/Colab/local machine) continue the project
without the original chat. Read it top to bottom once; then use §7 to set up and §10 to pick the next task.

* **Repository:** https://github.com/hbasired/defectspotter
* **Working branch:** `claude/optimistic-mccarthy-q7e5mh` (merge into `main` when reviewed — see §10, task P0-1)
* **Planned public demo:** Hugging Face Space `bhkr0901/defectspotter` (not deployed yet — §9)
* **Last updated:** 2026-11-19 (end of the first build session)

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
| O9 | Public demo URL | Hugging Face ZeroGPU Space live | ⏳ needs the owner's HF write token (§9) |

**In scope:** MVTec AD bottle / screw / carpet (+ metal_nut for few-shot), research/teaching use, CPU-friendly inference.
**Out of scope:** commercial use of MVTec data, certified production control, PLC/MES integration, multi-camera fusion,
authentication / persistent multi-user storage (the review queue lives in the browser session).

## 3. What was built (architecture)

```
data (MVTec AD, pinned HF mirror)
  └─ scripts/train.py → defectspotter/pipeline.py (stages)
       index → eda → synthetic → patchcore → vae → mean → diffusion → calibrate → evaluate → latency → drift → few_shot → examples → summary
       writes: artifacts/ (deployable models) · reports/ (CSV/JSON/PNG evidence) · examples/ (demo images)
app.py → ui/app.py (Gradio) → defectspotter/runtime.py (ONNX Runtime + NumPy only, no PyTorch)
```

| Component | Key design decision | Why |
|---|---|---|
| Feature extractor | ResNet-18 `layer2+layer3`, 3×3 avg aggregation, 224 px, ONNX (10.6 MB) | backbone study: best accuracy inside the 10–20 MB budget (MobileNetV3 0.85 mean AUROC, EffNet-B0 0.91, **ResNet-18 0.93**, WRN-50 0.94 at 95 MB) |
| Memory bank | **greedy k-center coreset** (JL projection to 128-d), float16 | the original notebook used random sampling; greedy keeps rare-but-normal patches |
| Hyper-parameters | chosen on held-out good + **synthetic** defects only; objective = 0.4·img AUROC + 0.3·TPR@≤1%FPR + 0.3·pixel AUROC | synthetic AUROC alone saturates at 1.0 and cannot rank configs |
| VAE | 4-stage conv encoder → 1×1 squeeze → 128-d latent; L1 + 0.5(1−SSIM) + β·KL; ONNX ≈7.4 MB | small, sharper than MSE; no flips for screws (thread handedness) |
| Baseline | ECC alignment (none/translation/euclidean/affine picked on synthetic data) + per-pixel z-score | transparent classical comparison |
| Score scale | 0–100, 25 = median held-out good, 75 = median synthetic defect | label-free, interpretable for operators |
| Pixel scale | 0 = median good pixel, 50 = 99.9th pct good pixel, 100 = typical synthetic-defect pixel | localised, non-saturating heat maps |
| Decision | PASS < t−b ≤ CHECK < t+b ≤ REJECT, (t,b) minimise expected cost s.t. CHECK ≤ 10% | mirrors a real line: prevalence 1%, escape 400, false reject 2, CHECK 0.30 (illustrative units) |
| Health gate | brightness/contrast/sharpness z-score vs qualified images, \|z\|>4 → no automatic decision (CHECK) | lighting drift otherwise causes mass false rejects |
| Drift | EWMA (λ=0.2, 3σ, 2-in-a-row) on winsorised scores and health signals + rolling PSI | SPC practice; health signals lead for lighting, score chart needed for geometry |
| Few-shot | memory bank from k good images; threshold from leave-one-image-out scores | PatchCore needs no network training |
| Diffusion defects | SD-1.5 inpainting + LCM-LoRA (4 steps, guidance 1.0), procedural masks, masked composite | GenAI synthetic defects that run even on CPU |

Splits (seed 42): train/good → 80% `model_train_good` + 20% `calibration_good`; test → per defect type 35%
`oracle_calibration` (real-label threshold, comparison only) + 65% `operating_test`. Real defects never train a model or
choose the default threshold.

## 4. Repository map

| Path | Purpose |
|---|---|
| `app.py` | entry point (local / Colab / HF ZeroGPU: imports `spaces` first, registers a `@spaces.GPU` self-test) |
| `ui/` | `app.py` (10 tabs + API), `charts.py` (Plotly, colour-blind-validated palette, single y-axis), `story.py` (presentation HTML, numbers read from reports), `store.py`, `theme.py` (light theme, forced even in OS dark mode) |
| `defectspotter/` | `data`, `synthetic`, `features`*, `patchcore`, `vae`*, `baseline`, `metrics`, `decision`, `drift`, `viz`, `runtime`, `pipeline`* (*need PyTorch, training only) |
| `scripts/` | `download_data.py`, `train.py`, `generate_diffusion_defects.py`, `backbone_study.py`, `deploy_hf.py`, `build_report.py`, `summarize_results.py` |
| `artifacts/` | `manifest.json`, `feature_extractor.onnx`, `<product>/{calibration.json, patchcore_bank.npy, vae.onnx, mean_baseline.*}` |
| `reports/` | every metric/table/figure (`metrics.csv`, `operating_metrics.csv`, `threshold_comparison.csv`, `scores.json`, `drift_*.csv`, `few_shot.csv`, `backbone_study.csv`, `figures/`, `DefectSpotter_report.html`) |
| `examples/` | down-scaled MVTec test images + masks for the demo, synthetic and diffusion galleries, new-product samples |
| `DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb` | Colab: clones the repo, runs each stage with inline charts, launches the app, deploys |
| `DefectSpotter_HuggingFace_Deployment_Guide.md` | local / Colab / HF / Claude deployment details |
| `docs/` | `MODEL_CARD.md`, `DATA_CARD.md`, `screenshots/` |
| `tests/` | `test_core.py` (unit), `test_app.py` (integration on shipped artifacts) |

## 5. Results snapshot

<!-- HANDOFF-RESULTS:START -->
Filled in from `reports/` at the end of the run (see README "Results" and `python scripts/summarize_results.py`).
<!-- HANDOFF-RESULTS:END -->

## 6. Environment facts (as tested)

* Python 3.13 (works 3.10+), torch 2.15 CPU, torchvision 0.29, onnxruntime 1.32, gradio 6.32, plotly 7.1 (plotly.js 4.1.1),
  numpy 2.5, pandas 3.1, OpenCV 5.1, diffusers 0.43, transformers 5.23, peft 0.22.
* Hardware used: 4 vCPU, 15 GB RAM, no GPU. Full pipeline ≈ 1 h; VAE ≈ 3.5 s/epoch/product; diffusion ≈ 30–60 s per
  image on CPU with LCM (seconds on a GPU).
* Data: `foersben/mvtec-ad` @ `c75b39616f84db43677bcc8228caaafaf5096d7f` (an earlier pin pointed at a partial upload).
* Hugging Face rules (Nov 2026, official docs): free accounts (verified e-mail, > 30 days old) may host **2 Gradio Spaces on
  ZeroGPU**; CPU-basic Gradio/Docker Spaces need PRO; static Spaces are free; Streamlit is only available via Docker.
  ZeroGPU supports Python 3.10.13 / 3.12.12 and PyTorch 2.8–2.13; needs ≥ 1 `@spaces.GPU` function at start-up.

## 7. Set up from scratch (new machine, new cloud account or new Claude session)

```bash
git clone https://github.com/hbasired/defectspotter.git && cd defectspotter
git checkout claude/optimistic-mccarthy-q7e5mh           # until merged into main
python -m venv .venv && source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt                            # app only (no torch)
python app.py                                              # http://127.0.0.1:7860
pip install -r requirements-dev.txt && pytest              # unit + integration tests
```

Re-train (only needed after changing the method or data):

```bash
pip install -r requirements-train.txt
python scripts/download_data.py --dest data/mvtec_ad
python scripts/train.py --data data/mvtec_ad --work-dir ../defectspotter_work           # all stages
python scripts/train.py --data data/mvtec_ad --stages calibrate evaluate summary         # re-run selected stages
pip install -r requirements-diffusion.txt
python scripts/generate_diffusion_defects.py --data data/mvtec_ad --per-category 8 --steps 4 --guidance 1.0
python scripts/train.py --data data/mvtec_ad --stages diffusion calibrate evaluate summary
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

## 9. Deployment status and how to finish it

* Not yet deployed. Run `python scripts/deploy_hf.py --space bhkr0901/defectspotter` with `HF_TOKEN` set (write scope).
  The script stages the app + artifacts (~20 MB), writes the Space card (Gradio 6.32, Python 3.12, ZeroGPU), uploads and prints
  `https://huggingface.co/spaces/bhkr0901/defectspotter` and `https://bhkr0901-defectspotter.hf.space`.
* If ZeroGPU is refused (account < 30 days or unverified e-mail) the Space is created on default hardware and will not start
  without PRO; alternatives: Colab `share=True` link (temporary), local `python app.py --share`, or host
  `reports/DefectSpotter_report.html` as a free **static** Space / claude.ai page (presentation only, no live inference).
* After deploying: open the Space, wait for the build, run one sample per product on the Inspect tab, check `/health`.

## 10. Next steps (prioritised backlog)

**P0 — finish & ship**
1. Review the branch and merge `claude/optimistic-mccarthy-q7e5mh` into `main` (open a PR).
2. Deploy the Space (§9) and smoke-test it; add the live URL to `README.md` and this file.
3. Rehearse the demo script (§8); optionally record a 2-minute screen capture as a fallback.

**P1 — improve results**
4. Carpet is the weakest product with the 11 MB backbone (backbone study: WRN-50 0.97 vs ResNet-18 0.86 on carpet).
   Options: texture-specific higher resolution, ResNet-34 / EfficientNet features for textures, or WRN-50 fp16 for carpet only
   (+45 MB). Select with the synthetic-validation objective, never with test labels.
5. Generate more diffusion defects on a GPU (e.g. 50 per product, more prompts) and re-run
   `--stages diffusion calibrate evaluate summary`; report whether diffusion closes the synthetic-vs-real threshold gap.
6. Test robustness on real phone photos of bottles/screws (domain shift) and document failure modes.
7. VAE: try a perceptual (feature) loss or a spatial latent; report the VAE vs PatchCore gap per defect type.

**P2 — productionise / extend**
8. Persist the review queue and audit log (SQLite / HF Dataset) with user identity; authentication on the Space.
9. MVTec AD 2 (lighting variations in the test set) to validate the drift story on real lighting changes.
10. Continuous monitoring page fed by real inspection logs (EWMA/PSI per shift, CHECK-rate alarm).
11. GitHub Actions CI running `pytest` on every push; pre-commit lint.
12. Optional in-browser demo (ONNX Runtime Web) for a fully static deployment.

## 11. Known issues and limitations

* Synthetic defects are easier than some real ones; thresholds can sit higher than the real-label oracle (see
  `reports/threshold_comparison.csv`).
* Down-scaling to 224 px hides very small defects (many MVTec defects cover < 1% of the image).
* Geometric camera shifts are not caught by brightness/sharpness health signals — rely on the score chart for those.
* The review queue is per browser session (no persistence) — fine for a demo, not for production.
* Score = ranking, not probability; heat map = visual deviation, not root cause.
* MVTec AD and everything derived from it (banks, VAEs, examples) are CC BY-NC-SA 4.0: non-commercial only.

## 12. Glossary

**FRR** false-reject rate (good part → REJECT) · **Escape rate** defective part → PASS · **CHECK / review load** share of
parts sent to a human · **AUROC** ranking quality (1 = perfect) · **Pixel AUROC / AUPRO@30%** localisation quality (AUPRO
weights every defect region equally) · **Coreset** small subset that covers the memory bank · **EWMA** exponentially
weighted moving average control chart · **PSI** population stability index (< 0.1 stable, 0.1–0.25 warn, > 0.25 alert).
