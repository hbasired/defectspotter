# DefectSpotter — the complete guide

*Everything in the project explained step by step: the problem, the data, every model and why it was chosen, every
concept (CNN, PatchCore, coreset, VAE, KL divergence, diffusion, AUROC, EWMA, …), every important parameter, how to
run it, how to present it.*

Live app: **https://huggingface.co/spaces/bhkr0901/defectspotter** · code: **https://github.com/hbasired/defectspotter**

---

## 1. The problem in one minute

A quality inspector at the end of a production line must stop **defective** parts without throwing away **good**
ones. Two errors cost money in opposite directions:

* **Escape** — a defective part passes and reaches the customer (complaints, warranty, recalls). Expensive.
* **False reject** — a good part is scrapped or reworked (material, throughput). Cheap per part, but frequent.

A normal "defect classifier" needs hundreds of labelled images **per defect type**. Factories rarely have them:
defects are rare, varied and keep changing — a new scratch pattern the classifier never saw simply passes.

**DefectSpotter turns the problem around: it learns only what *good* parts look like** and flags anything unusual
(*anomaly detection*). It returns an **anomaly score**, a **heat map** of where the anomaly is, and a three-way
decision: **PASS** (ship), **CHECK** (a human inspector decides), **REJECT** (quarantine). Not a single defect image is
needed for training *or* for choosing the decision threshold.

## 2. Quick start (step by step)

**Use the live app (nothing to install).** Open the Space link above. The tabs go from the presentation (Overview →
Data → Models & training → Results → Threshold & cost → Drift & new lots → New product → Guide) to the
**🎬 Live demo** (last tab). Upload one of the images in `demo_images/` or click a sample.

**Run it on your computer.**

```bash
git clone https://github.com/hbasired/defectspotter.git
cd defectspotter
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                        # app only: NumPy, OpenCV, ONNX Runtime, Gradio, Plotly
python app.py                                          # open http://127.0.0.1:7860
```

The trained models (`artifacts/`), the evidence the dashboard plots (`reports/`) and sample images (`examples/`) are
in the repository, so the app works immediately — no GPU, no PyTorch.

**Re-train everything** (CPU ≈ 2–2.5 h on 4 cores, a GPU is ≈ 5× faster):

```bash
pip install -r requirements-train.txt                  # + PyTorch, torchvision, ONNX export
python scripts/download_data.py --dest data/mvtec_ad   # MVTec AD from a pinned Hugging Face mirror (1.3 GB)
python scripts/train.py --data data/mvtec_ad --stages index eda synthetic
pip install -r requirements-diffusion.txt              # optional: Stable Diffusion defects
python scripts/generate_diffusion_defects.py --data data/mvtec_ad --per-category 12
python scripts/train.py --data data/mvtec_ad --vae-epochs 120 \
       --stages patchcore vae fvae mean severity diffusion calibrate evaluate requalify latency drift few_shot examples summary
python scripts/summarize_results.py --write            # refresh the result tables in README / handoff
python scripts/build_report.py                         # static HTML report
pytest                                                 # unit + integration tests
```

Every stage caches its inputs in `../defectspotter_work`, and the long stages (PatchCore search, VAE training) are
resumable per product, so a stage can be re-run alone after a change.

**Google Colab.** Open `DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb`, run the cells top to bottom (it clones this
repository, runs every stage with inline charts, launches the dashboard and can deploy to Hugging Face).

**Deploy to Hugging Face.** `HF_TOKEN=<write token> python scripts/deploy_hf.py --space <user>/defectspotter`
(details in `DefectSpotter_HuggingFace_Deployment_Guide.md`).

## 3. The data

**MVTec AD** (MVTec Software GmbH; Bergmann et al., IJCV 2021) is the standard benchmark for industrial anomaly
detection: 5,354 high-resolution images of 15 products. The **training set contains only good parts**; the **test
set** contains good parts and real defects, each defect with a pixel-accurate **ground-truth mask**.

| product | why it was chosen | train (good) | test good | test defective | defect types |
|---|---|---|---|---|---|
| bottle | glass object, fixed top view — the "easy" case | 209 | 20 | 63 | broken large, broken small, contamination |
| screw | small metal object at **random orientation** — the hard object | 320 | 41 | 119 | manipulated front, scratch head, scratch neck, thread side, thread top |
| carpet | woven **texture** — no object shape to rely on | 280 | 28 | 89 | color, cut, hole, metal contamination, thread |
| metal nut | held back as a "new product" for few-shot onboarding | 220 | 22 | 93 | bent, color, flip, scratch |

Licence: **CC BY-NC-SA 4.0** — research and teaching only, no commercial use. Downloaded from the Hugging Face mirror
`foersben/mvtec-ad`, pinned to one commit so every run gets byte-identical files.

### 3.1 The split (no leakage)

| role | what it is | used for |
|---|---|---|
| ① `model_train_good` | 80% of the training goods | building the memory bank, training the VAEs |
| ② `calibration_good` | 20% of the training goods, never seen by a model | the score scale, thresholds, hyper-parameter choice (together with synthetic defects) |
| ③ `oracle_calibration` | 35% of each test type | the "what if we had labels" threshold (comparison only) and the good parts of the *current lot* for re-qualification |
| ④ `operating_test` | 65% of each test type | **every** reported false-reject / escape / CHECK number — never used for any choice |

Real defects never train a model and never choose the deployed threshold — exactly the situation of a factory before
its first defect appears.

### 3.2 What the exploratory data analysis (EDA) showed

* **Tiny defects dominate**: most defect masks cover less than 1% of the image → localisation metrics matter, and
  down-scaling can hide defects.
* **Acquisition is tightly controlled** (small brightness/contrast spread) — what makes good-only learning possible,
  and why a lamp or focus change is dangerous.
* **Test captures differ slightly from training captures**: carpet test images are ~4 grey levels brighter and come
  from a different roll; screw test images are darker. Section 9.4 shows how this "lot change" is handled.
* Test sets are ≈70% defective, unlike a real line (≈1%), so all rates are reported per class and costs are
  re-weighted to a 1% defect rate.

## 4. The pipeline at a glance

```
good parts ──► frozen ResNet-18 (ONNX) ──► patch descriptors ──► PatchCore memory bank ──► distance map ──► score
            └► feature-space VAE (trained on the descriptors) ──► reconstruction error map ──► score
            └► pixel VAE (trained on images, v1 comparison) ──► reconstruction error map ──► score
            └► aligned mean image ± std (classical baseline) ──► z-score map ──► score
held-out good parts + synthetic defects ──► 0–100 score scale + cost-optimal PASS / CHECK / REJECT thresholds
inspection ──► camera-health gate ──► decision ──► CHECK → human review queue + audit log ──► drift monitoring
```

Pipeline stages (`python scripts/train.py --stages …`), in order:

| stage | what it does | main outputs |
|---|---|---|
| `index` | finds every image/mask, makes the leakage-free split | `reports/splits.csv` |
| `eda` | per-image statistics, defect sizes/locations, montages | `reports/image_stats.csv`, figures |
| `synthetic` | paints cut-paste / scratch / stain defects on held-out good parts | synthetic images + masks |
| `patchcore` | label-free search of the per-product PatchCore recipe, builds the memory bank | `artifacts/<p>/patchcore_bank.npy`, `reports/hpo_patchcore.csv` |
| `vae` | trains the pixel VAE (v1) | `artifacts/<p>/vae.onnx` |
| `fvae` | trains the feature-space VAE | `artifacts/<p>/fvae.onnx` |
| `mean` | classical baseline (alignment chosen on synthetic data) | `artifacts/<p>/mean_baseline.*` |
| `severity` | replays every synthetic defect at 50% and 25% strength | scores |
| `diffusion` | scores the Stable-Diffusion defects (if generated) | scores |
| `calibrate` | score scales, ensemble weight, thresholds; exports the ONNX backbone | `artifacts/<p>/calibration.json` |
| `evaluate` | all metrics on the test split | `reports/metrics.csv`, `operating_metrics.csv`, … |
| `requalify` | lot-shift test with current-lot good parts; re-qualifies where needed | `reports/requalification.csv`, `artifacts/<p>/requalified/` |
| `latency` | end-to-end time per part on CPU | `reports/latency.csv` |
| `drift` | lighting / focus / camera stress test, 200-part stream with control charts | `reports/drift_*.csv` |
| `few_shot` | new-product experiment (metal nut) | `reports/few_shot.csv` |
| `examples`, `summary` | demo images for the app, `summary.json` | |

## 5. The backbone: a frozen ResNet-18

### 5.1 Concepts

* **Convolutional neural network (CNN).** A stack of small learned filters slid over the image. Early layers respond to
  edges and colours, middle layers to textures and small shapes, deep layers to object parts. The output of a layer is
  a *feature map*: a grid where every cell holds a vector describing the image patch under it.
* **ResNet (residual network).** A CNN built from *residual blocks* that add their input to their output
  (`y = x + F(x)`); this "shortcut" lets very deep networks train. **ResNet-18** has 18 weight layers grouped into
  four stages (`layer1`–`layer4`) with 64, 128, 256 and 512 channels; each stage halves the resolution (stride 4, 8,
  16, 32 relative to the input).
* **Transfer learning / pre-training.** The weights come from training on **ImageNet-1k** (1.28 million everyday photos,
  1,000 classes). Those generic visual features transfer remarkably well to industrial images.
* **Frozen = not fine-tuned.** We do **not** change a single backbone weight and we do **not** attach a classification
  head: there are no defect labels to train one, and fine-tuning on good parts only would teach it nothing useful
  (it could even collapse the features). The backbone is used as a fixed "feature camera".

### 5.2 What exactly is used

| item | value |
|---|---|
| weights | torchvision ResNet-18, ImageNet-1k |
| layers kept | `conv1 → layer1 → layer2 → layer3` (2.78 M parameters); `layer4`, global pooling and the classifier head are removed (too ImageNet-specific, not needed) |
| feature maps | `layer2` (128 channels, stride 8 — fine texture) and `layer3` (256 channels, stride 16 — object parts), `layer3` upsampled to `layer2`'s grid and concatenated → **384 numbers per patch** |
| local aggregation | output 1: **3×3 neighbourhood average** (PatchCore default, robust to tiny shifts); output 2: **raw descriptors** (keep fine texture) |
| input | resized to 224 px (28 × 28 patch grid) or 288 px (36 × 36), normalised with the ImageNet mean/std |
| export | one **ONNX** file (10.6 MB), dynamic batch and resolution, verified against PyTorch (max error ≈ 2·10⁻⁶) |

**Why ResNet-18?** A backbone study ran PatchCore on four backbones: MobileNetV3-Small (0.85 mean AUROC),
EfficientNet-B0 (0.91), **ResNet-18 (0.93)** and WideResNet-50 (0.94 — but a 95 MB model). ResNet-18 is the most
accurate model inside the 10–20 MB deployment budget. **ONNX** (Open Neural Network Exchange) lets the app run the
network with the light ONNX Runtime, without PyTorch.

## 6. PatchCore — a memory of normal patches (the deployed detector)

### 6.1 Idea

A part is normal if **every small patch of it looks like some patch of some good part**. So:

1. **"Training" (no gradient descent):** run the backbone over all good training images and keep every patch descriptor
   in a **memory bank**.
2. **Inference:** for every patch of a new image, find its **nearest neighbour** in the bank (Euclidean distance). A
   defect patch is far from everything normal.
3. **Heat map:** the grid of distances, upsampled to the image size and smoothed with a Gaussian (σ = 2–4 px).
4. **Image score:** the largest patch distance (`max`) or the mean of the top 1–5% of the map (`top-k`, less sensitive
   to a single noisy patch).

### 6.2 Coreset — keeping the bank small without losing rare normal patches

Hundreds of good images × 784–1,296 patches = hundreds of thousands of vectors. A **greedy k-center coreset** keeps a
small subset that still *covers* all of them: start with one patch, then repeatedly add the patch **farthest from
everything kept so far**. Rare-but-normal details (a reflection seen in 2% of bottles) survive — random subsampling would
drop them. For speed, distances are computed after a random **Johnson–Lindenstrauss projection** to 128 dimensions
(which approximately preserves distances). Two engineering tricks: one greedy run serves every coreset ratio (its first
*k* picks *are* the size-*k* coreset), and very large sets are first reduced to a random 300 k candidate pool. Banks are
stored in float16 (half precision).

### 6.3 The per-product recipe and how it was chosen without labels

Every product gets its own settings, searched on **held-out good parts + synthetic defects** only:

| option | values searched | why it matters |
|---|---|---|
| input resolution | 224, 288 px | small defects need pixels; too fine a scale over-reacts to texture |
| feature aggregation | 3×3 average, raw | raw keeps fine weave detail (carpet) |
| pose normalisation | off / on (elongated parts only) | rotates every screw to the same pose (Section 6.4) |
| rotated bank | 1 or 4 orientations of each good part (objects) | covers random orientations without alignment |
| coreset ratio | 1%, 5%, 10% | memory vs coverage |
| blur σ, score mode | 2/4 px; max, top-1%, top-5% | smoothness and robustness of the score |

**Objective** (higher is better): `0.4 × image AUROC + 0.3 × TPR at ≤ 1% false rejects + 0.3 × pixel AUROC`, where
the "defects" are full-strength synthetic defects **plus their 50%-strength replays** (full-strength defects alone are
too easy — every setting scores ≈ 1.0). **Parsimony rule:** among all settings within 0.005 of the best objective
(differences that small are sampling noise), deploy the one with the cheapest nearest-neighbour search. The deployed
recipes are shown in the app (Models & training) and in `reports/hpo_patchcore.csv` (column `selected`).

### 6.4 Pose normalisation (screws)

CNN features are not rotation-invariant, and screws lie on the tray at any angle, so a memory bank would have to contain
every orientation of every thread. Classic machine vision fixes the pose first: the part mask (Otsu threshold on the
colour distance to the background) gives the **principal axis** (PCA of the mask pixels); the image is rotated about the
part's centroid so that the axis is horizontal and the **wider end (the head) is on the left**, and the part is centred.
Rotation keeps the thread's handedness (a mirror flip would not). Heat maps are warped back to the original frame. Round
parts (bottle top view) have no axis and are never aligned.

## 7. Generative AI: variational autoencoders

### 7.1 Autoencoder and VAE — the concepts

* An **autoencoder** has an **encoder** that compresses the input into a small code (the *latent*) and a **decoder**
  that rebuilds the input from it. It is trained to make the reconstruction equal to the input.
* Trained **only on good parts**, it learns to rebuild *normal* appearance. Shown a defect, it rebuilds what a good part
  would look like there — so the **reconstruction error** lights up exactly where the defect is.
* A **variational** autoencoder (VAE) makes the latent random: the encoder outputs a mean μ and a log-variance log σ²,
  and a sample `z = μ + σ·ε` (ε ~ N(0, 1), the *reparameterisation trick*) is decoded. A **KL-divergence** term
  `KL(q(z|x) ‖ N(0, I)) = −½ Σ(1 + log σ² − μ² − σ²)` keeps the latent distribution close to a standard normal. This
  regularises the code (smooth, no "memorising"), and makes the model generative (you could sample new good-looking
  parts). The training loss is `reconstruction error + β · KL` (β small, 0.001 here, so reconstructions stay sharp).
  For scoring we decode the mean μ (deterministic).

### 7.2 Pixel VAE (v1 — kept for comparison and its reconstruction view)

| item | value |
|---|---|
| input | RGB image resized to 128 × 128, values in [0, 1] |
| encoder | 4 stages of (stride-2 4×4 conv → GroupNorm → SiLU → 3×3 conv → GroupNorm → SiLU), 32 → 64 → 128 → 128 channels, 1×1 "squeeze" to 32 channels (8 × 8) |
| latent | 128 numbers (fully connected from the 32 × 8 × 8 map) |
| decoder | mirror image with transposed convolutions, sigmoid output |
| loss | L1 + 0.5 · (1 − SSIM) + 0.001 · KL — **SSIM** (structural similarity) compares local brightness, contrast and structure, so the model is not rewarded for blurry averages |
| training | 120 epochs, AdamW (weight decay 1e-5), one-cycle learning rate up to 1e-3, batch 32; augmentation: horizontal flips (not for screws — it would mirror the thread), 90° rotations, ±4% brightness; the checkpoint with the lowest validation loss is kept |
| anomaly map | 0.5 · |input − reconstruction| + 0.5 · (1 − SSIM map), Gaussian σ 2; score = mean of the top 1% |
| size | ≈ 2 M parameters, 7.4 MB ONNX per product |

It squeezes a whole image through 128 numbers, so it reconstructs a **blurry average part**: the error is large
everywhere, and defects barely stand out (screw AUROC ≈ 0.5 — random).

### 7.3 Feature-space VAE (v3 — the generative detector)

The better VAE reconstructs **what the backbone sees** instead of raw pixels: the 384 × 28 × 28 grid of patch
descriptors (224 px input, 3×3-averaged — the same features PatchCore uses).

| item | value |
|---|---|
| input | backbone feature map, normalised per channel with the training mean/std |
| encoder | 1×1 conv 384 → 256 · 3×3 stride-2 conv 256 → 128 (14 × 14) · 3×3 stride-2 conv 128 → 128 (7 × 7); GroupNorm + SiLU |
| latent | **spatial**: 32 channels × 7 × 7 (μ and log σ² per position) — keeps *where* things are |
| decoder | 1×1 conv → two 4×4 stride-2 transposed convolutions back to 28 × 28 → 1×1 conv to 384 channels |
| loss | mean-squared error on the (normalised) features + 0.001 · KL |
| training | 60 epochs, AdamW, one-cycle learning rate up to 2e-3, batch 16, feature-map flips (not for screws); best validation checkpoint |
| anomaly map | per-patch squared reconstruction error (computed inside the ONNX graph), upsampled, Gaussian σ 4; score = top-1% mean |
| size | 1.44 M parameters, ≈ 5.5 MB ONNX per product, ≈ 2 min training per product on a CPU |

**Why it works better:** descriptors are compact and semantic (a scratch is "unusual texture", not millions of pixel
values), and the spatial latent keeps the layout, so normal patches are reconstructed accurately while defect patches —
which the decoder never learned to produce — are not. Its AUROC is far above the pixel VAE (see Results).

### 7.4 Ensemble

`w · PatchCore score + (1 − w) · feature-VAE score`, with `w` chosen on held-out good + synthetic (full and 50%)
defects, scanning from PatchCore-only downwards so ties keep the stronger single detector.

## 8. Classical baseline

Every image is aligned to a median good reference with OpenCV's **ECC** algorithm (enhanced correlation coefficient;
none / translation / rigid / affine chosen on synthetic data), then compared pixel by pixel with the **mean and
standard deviation** of the aligned good parts: `z = |pixel − mean| / std` (blurred). Transparent and fast — and it
shows exactly where learned features are needed (it fails on rotating screws and on textures).

## 9. From a score to a decision (calibration)

### 9.1 Synthetic defects (no real defects needed)

On every held-out good part: **cut-paste** (a patch of the same part, rotated and recoloured, pasted elsewhere — the
CutPaste method), **scratch** (a thin jittered line, darker or brighter than the surface) and **stain** (an irregular
noise-shaped blob with a colour shift), each with an exact mask, kept on the part by the part mask. Every synthetic
defect is replayed at **50% and 25% strength** (blended with the original) — faint defects mark the *detection limit*,
like an inspection specification. A synthetic set only influences the default threshold if the detector separates it
from good parts (synthetic AUROC ≥ 0.9, a label-free *capability check*).

**Diffusion-refined defects (GenAI).** A diffusion model generates images by learning to remove noise step by step.
**Stable Diffusion 1.5** with **SDEdit** (image-to-image: add partial noise — *strength* 0.45 — to a crop around a
procedural defect and let the model denoise it, guided by a product-specific prompt such as "a deep scratch and a dented
damaged thread on a steel screw, macro inspection photo") keeps the defect's structure but gives it photographic texture.
An **LCM-LoRA** (latent-consistency LoRA: a small low-rank adapter that distils the model to work in a handful of steps)
makes this feasible on a CPU: 6 steps, guidance 1.5, ≈ 17 s per image. Only the masked region is composited back.

### 9.2 Score scale and three-way decision

* Raw distances are meaningless to an operator, so every detector's score is mapped to a **0–100 scale**:
  **25 = median held-out good part, 75 = median synthetic defect**. Pixels get a similar scale (50 = more unusual than
  99.9% of good pixels).
* Decision with a threshold *t* and a CHECK half-band *b*: **PASS** if score < t − b, **CHECK** if t − b ≤ score < t + b,
  **REJECT** if score ≥ t + b.

### 9.3 Choosing (t, b): the cost model

Minimise the expected cost per part

`(1 − p) · FRR · C_false-reject + p · escape rate · C_escape + review rate · C_check`

with defect prevalence *p* = 1%, `C_escape` = 400, `C_false-reject` = 2, `C_check` = 0.30 (illustrative units), on
held-out good parts + capability-checked synthetic defects. Two refinements: the rates are estimated on
**kernel-smoothed** score distributions (Gaussian kernels with Silverman's bandwidth — with ~50 good parts a raw
false-reject rate moves in 2% steps), and the CHECK band is planned to use at most **50% of the inspector capacity**
(10% of parts) — headroom for drift, as in any capacity planning. The threshold lab in the app recomputes this live.

### 9.4 New production lots: re-qualification

A new lot (another carpet roll, a new screw batch) can look slightly different while still being good — the model then
scores good parts higher and sends them to CHECK. **Re-qualification** needs only a handful of **good** parts of the
new lot:

1. **Lot-shift test:** a one-sided **Mann–Whitney U test** (a rank test: do the new good parts score higher than the
   qualification's held-out good parts?). p < 0.01 → lot change.
2. If changed: a **10% greedy coreset of the new parts' patches joins the memory bank** (old lots stay normal), their
   **leave-one-out** scores (each scored against the bank plus the *other* new parts) become the good reference,
   **synthetic defects are generated from the new parts**, and the score scale and thresholds are re-fitted exactly as
   in 9.3. No defect image is needed.

On MVTec, the test captures play the "current lot" (the oracle split's good parts are the re-qualification sample, the
operating split measures the effect).

## 10. Metrics

* **Image AUROC** (area under the ROC curve): the probability that a random defective part scores higher than a random
  good part. 1.0 = perfect ranking, 0.5 = guessing. Threshold-independent. Reported with a **95% bootstrap confidence
  interval** (resample the test set 1,000 times, stratified by class).
* **ROC curve:** true-positive rate (defects caught) vs false-positive rate (good parts flagged) as the threshold moves.
* **Average precision (AP):** area under the precision–recall curve.
* **Pixel AUROC:** the same for pixels — do pixels inside the ground-truth masks score higher than normal pixels?
* **AUPRO @ 30% FPR:** per-region overlap of the thresholded heat map with each defect, averaged up to 30% false-positive
  pixels, so a small defect counts as much as a large one.
* **False-reject rate (FRR):** good parts sent to REJECT. **Escape rate:** defective parts that PASS.
  **CHECK load:** share of all parts sent to a human. **Confusion matrix:** rows = truth (good / defective), columns =
  decision (PASS / CHECK / REJECT).
* **Expected cost per 1,000 parts:** the cost model above at 1% prevalence. **Latency:** end-to-end milliseconds per
  part on a 4-core CPU.

## 10b. Results (held-out MVTec test images)

| | bottle | screw | carpet | mean |
|---|---|---|---|---|
| **PatchCore image AUROC** (deployed) | 1.000 | **0.997** | 0.940 | **0.979** |
| PatchCore pixel AUROC | 0.980 | 0.996 | 0.983 | 0.986 |
| PatchCore AUPRO @ 30% | 0.945 | 0.972 | 0.931 | 0.949 |
| **Feature-space VAE** image AUROC (v3) | 0.997 | 0.798 | 0.903 | 0.899 |
| Pixel VAE image AUROC (v1) | 0.902 | 0.477 | 0.671 | 0.683 |
| Mean-image baseline image AUROC | 0.975 | 0.277 | 0.435 | 0.562 |

**Deployed decisions on the operating test split** (thresholds chosen without real defects):

| product | model | false rejects | escapes | parts sent to a human (CHECK) |
|---|---|---|---|---|
| bottle | qualification lot | 0% | 0% | 0% |
| screw | qualification lot (pose-normalised) | 0% | 6.4% | 0.2% |
| carpet | **re-qualified for the current lot** (10 good parts) | 0% | 8.8% | 5.7% (was 49.6% before re-qualification) |

How the project got here (image AUROC, PatchCore): v1 bottle 1.000 / screw 0.935 / carpet 0.880 → v2 (per-product
recipe) 1.000 / 0.983 / 0.940 → v3 (pose-normalised screws) 1.000 / 0.997 / 0.940. The label-free threshold lands within
1.5–3.5 points of the threshold a plant would choose with labelled defects. Note: the lot-shift test found no change for
screw (p = 0.86), so its qualification model stays deployed even though re-qualifying it would also have lowered its
escapes on the test split — the decision rule is fixed in advance, not picked from test results.

## 10c. Real-world numbers: one production shift

Rates are easier to judge as counts. The app's Overview turns the deployed operating points into **one 8-hour shift** of
a realistic inspection station (all values editable in *Threshold & cost*):

| assumption | value | source / reasoning |
|---|---|---|
| line speed | 2,500 parts/hour → 20,000 parts per shift | a mid-speed station |
| defect rate | 1% → 200 defective parts per shift | typical first-pass yield |
| human check of a flagged part | 36 s → 100 checks per inspector-hour | pick the part, look with the heat map, decide |
| inspector cost | 30 USD/hour fully loaded → **0.30 USD per check** | BLS median pay for quality-control inspectors 22.82 USD/h (2024) + ~30% benefits/overhead |
| inspector hit rate | 80% of the defects they are shown | industry-average visual-inspection hit rate (See, *Human Factors* 2015) |
| false reject | 2 USD | value of a scrapped mid-value part |
| escape | 400 USD | 1-10-100 rule of cost of quality (Labovitz, Chang & Rosansky 1992): a defect costs far more at the customer; a customer complaint ≈ 1,000× the inspection cost |

One shift at the deployed thresholds:

| product / model | parts sent to a human | inspectors needed | defects escaping | cost per shift |
|---|---|---|---|---|
| bottle | 10 | 0.0 | 2.0 | 783 USD |
| screw | 44 | 0.1 | 21.5 | 8,628 USD |
| carpet, trained-lot model (before re-qualification) | 9,925 | **12.4** | 15.4 | 9,153 USD |
| carpet, re-qualified for the current lot (deployed) | 1,132 | **1.4** | 23.9 | 9,883 USD |
| 100% manual inspection (status quo) | 20,000 | 25 | 40 | 22,000 USD |

**Was lowering the human workload right?** Yes. Before re-qualification carpet would have needed about 12 inspectors per
shift; a station has 2–3, so most flagged parts would simply never be looked at (and an inspector who sees good parts
all day stops looking carefully). Re-qualification brings it to 1.4 inspectors. The honest price is about 8 more
defective carpet sections escaping per shift (24 instead of 15) — still far better than 100% manual inspection, which
needs 25 inspectors and misses 40 defects per shift. If a plant can staff more inspectors, raising *Inspector capacity*
in the threshold lab widens the CHECK band and buys those escapes back.

## 11. Responsible operation and monitoring

* **Human in the loop:** every CHECK goes to the review queue; the inspector's verdict and note go into an audit log.
  The model is never silently retrained from these decisions.
* **Camera-health gate:** brightness, contrast and sharpness of every image are compared with the qualified images
  (z-scores); beyond |z| > 4 no automatic decision is taken (PASS/REJECT → CHECK).
* **Drift monitoring:** an **EWMA control chart** (exponentially weighted moving average, λ = 0.2, ±3σ limits, alarm
  after 2 points in a row) on the score stream and on the health signals, and the **PSI** (population stability index:
  < 0.1 stable, 0.1–0.25 watch, > 0.25 alarm) of the score distribution. Lighting drift is caught by the health signals,
  a bumped camera by the score chart.
* **Limitations:** the score is a ranking, not a probability; heat maps show visual deviation, not root cause;
  synthetic defects do not cover every real failure mode; MVTec AD is non-commercial.

## 12. Few-shot onboarding of a new product

Because PatchCore needs no network training, a new product line can be onboarded from a few good images: build the bank
from them, score each with the others (leave-one-out) to learn the good-score distribution, generate synthetic defects
from them to set the threshold. The metal-nut experiment reaches ≈ 0.98 image AUROC with 8 good images
(`reports/few_shot.csv`).

## 13. Live demonstration

The `demo_images/` folder contains images to upload during a presentation (see `demo_images/README.md` for what each
one should show). A 3-minute script:

1. **Overview** tab: the problem and the headline numbers (hero tiles).
2. **🎬 Live demo → Inspect a part**, product *Bottle*: upload `demo_images/bottle/defect_broken_large.png` → REJECT
   with the heat map on the break; upload a good bottle → PASS. Point at the score bar and the "Why?" explanation.
3. Product *Screw*: upload a scratched screw — PatchCore localises the scratch; compare all detectors.
4. Move the **threshold** slider: watch the decision change; ↺ restores the calibrated policy.
5. **Batch**: load the sample batch → decision tiles, score strip, CSV; CHECK parts appear in the **Review queue** —
   confirm or release one (audit log).
6. **Drift & new lots**: dim the lamp on a good part (model alone → REJECT, health gate → CHECK); in section C press
   *Load the sample current-lot good parts* → *Re-qualify* for carpet.

## 14. Glossary

**Anomaly detection** — learning normality and flagging deviations. **Backbone** — the pre-trained network that turns
images into features. **Feature map / descriptor** — the vector a CNN layer computes for each image patch.
**Memory bank** — the stored descriptors of good parts. **Coreset** — a small subset that covers a large set.
**Nearest neighbour** — the closest stored vector. **VAE** — variational autoencoder. **Latent** — the compressed code
inside an autoencoder. **KL divergence** — a measure of how different two probability distributions are.
**SSIM** — structural-similarity index. **SDEdit** — image editing with a diffusion model by noising and denoising.
**LoRA** — low-rank adaptation, a small add-on that changes a large model's behaviour. **ONNX** — portable model format.
**AUROC / AUPRO / AP** — see Section 10. **FRR** — false-reject rate. **Escape** — a defect that passes. **CHECK** —
human review. **EWMA / PSI** — see Section 11. **Mann–Whitney U** — a rank test comparing two samples.
**Re-qualification** — updating the reference of "good" for a new production lot.

## 15. FAQ

* **Is the backbone fine-tuned?** No. ResNet-18 keeps its ImageNet weights; no head is attached.
* **So what is trained?** The memory bank is *built* from good parts (no gradients); the two VAEs are trained from
  scratch on good parts; the baseline computes statistics; the calibration fits a score scale and thresholds.
* **Were real defects used anywhere?** Only for evaluation and for the comparison "oracle" threshold — never to train a
  model or set the deployed threshold. The test set was looked at to propose design options (like the backbone
  study); the choice within those options is label-free (see README).
* **Why PASS / CHECK / REJECT and not just good / bad?** Uncertain parts are cheaper to check by a human than to scrap
  or ship; the CHECK band absorbs the overlap between the score distributions.
* **What happens with a new lot or a camera change?** The camera-health gate and the control charts raise an alarm;
  a new lot is re-qualified from a handful of good parts (Section 9.4).
