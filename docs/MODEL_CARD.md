# Model card — DefectSpotter

## Intended use
Research and teaching demonstration of **good-only visual inspection**: a quality inspector at the end of a production
line uploads part images and receives an anomaly score, a heat map and a PASS / CHECK / REJECT decision. Every CHECK is
decided by a human.

**Out of scope:** autonomous safety-critical release decisions; commercial use of MVTec AD data; unqualified or drifting
cameras; treating the score as a calibrated defect probability.

## Components
| Component | Details | Size |
|---|---|---|
| Feature extractor | ImageNet ResNet-18 (torchvision weights), `layer2` + `layer3`, 3×3 local averaging, 224×224 input, ONNX opset 17 | ≈11 MB |
| PatchCore memory bank | greedy k-center coreset of good-part patch descriptors, float16, one per product | 1–15 MB per product |
| Convolutional VAE | 4-stage conv encoder, 1×1 squeeze, 128-d latent, mirrored decoder; L1 + SSIM + β·KL; ONNX | ≈9 MB per product |
| Mean-image baseline | ECC-aligned per-pixel mean / std of good parts | <1 MB per product |
| Ensemble | weighted PatchCore + VAE normalised scores and maps (weight chosen on synthetic data) | – |

## Training data
MVTec AD `bottle`, `screw`, `carpet` — **good training images only** (80% fit, 20% held out for calibration).
`metal_nut` is used only for the few-shot onboarding experiment. Licence CC BY-NC-SA 4.0.

## Calibration (no real defects)
Held-out good parts + procedural synthetic defects (cut-paste, scratch, stain) and, optionally, Stable-Diffusion
inpainted defects define the 0–100 score scale (25 = typical good, 75 = typical synthetic defect), tune hyper-parameters
and select the cost-optimal three-way thresholds under an explicit cost model (1% prevalence, escape ≫ false reject,
inspector time, review capacity ≤ 10%).

## Evaluation
MVTec test images never used for training or default threshold selection: image AUROC (+95% stratified bootstrap CI),
average precision, pixel AUROC, AUPRO@30% FPR, false-reject rate, escape rate, review load, expected cost per 1,000
parts, latency per part, synthetic-vs-real threshold gap, drift sensitivity and few-shot curves. See `reports/` and the
app's *Models* tab for the numbers of the shipped run.

## Known limitations
* Synthetic defects do not cover every real failure mode; validate critical modes on plant data.
* Down-scaling to 224 px can hide very small defects (many MVTec defects cover < 1% of the image).
* Texture products (carpet) are harder for the small backbone; WideResNet-50 is more accurate there but ~9× larger.
* Lighting, focus and camera-position changes raise scores on good parts — hence the camera-health gate and drift alarms.
* The VAE reconstructs blurry versions of parts, so subtle defects can hide in its own reconstruction error.
* Heat maps show visual deviation, not root cause.

## Responsible-AI controls
Human review of every CHECK with an audit log; no automatic decision on images outside the qualified acquisition range;
EWMA + PSI drift monitoring with alarms that trigger maintenance and re-qualification (never silent retraining).
