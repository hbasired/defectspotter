# DefectSpotter — Colab and Hugging Face deployment guide

## Deliverable architecture

The rebuilt notebook publishes two repositories under the configured Hugging Face account `bhkr0901`:

1. **`bhkr0901/defectspotter-artifacts`** — the model and evidence backend. It contains the verified ONNX feature extractor, per-category PatchCore memory banks, VAE checkpoints, aligned-mean baselines, calibration files, EDA tables, evaluation reports, drift evidence, saved figures, attributed examples, and run metadata.
2. **`bhkr0901/defectspotter`** — the Gradio frontend and inference service. It downloads the artifact repository at startup and exposes the interactive web application plus `/inspect_part`, `/inspect_batch`, and `/health` API endpoints.

This separation keeps the Space source small while allowing a completed experiment to replace the backend artifacts without rewriting the interface.

## What the new interface contains

The application is evidence-driven: its charts are reconstructed from the CSV and JSON reports produced by the current notebook run rather than from placeholder values.

- Project purpose, factory workflow, method explanations, and responsible-use boundary.
- MVTec AD provenance, selected categories, good-only training design, mirror used for download, and non-commercial license notice.
- Dataset composition and defect-type charts.
- Brightness, contrast, sharpness, resolution, and channel-statistic EDA.
- Defect-mask-area analysis.
- Saved notebook montages and qualitative localization examples.
- Upload, webcam snapshot, and clipboard image inspection.
- Approximately one-frame-per-second webcam streaming demonstration.
- PASS / CHECK / REJECT result, score gauge, heat-map overlay, numerical anomaly map, threshold contour, VAE reconstruction when available, and suspected-region table.
- Multiple-image batch inspection with a downloadable CSV and localized overlay gallery.
- Image AUROC, average precision, pixel AUROC, AUPRO, latency-versus-performance, operating-rate, expected-cost, threshold-gap, and artifact-size charts.
- Synthetic-only threshold versus labelled-real-defect oracle comparison.
- Drift score, PSI, brightness, contrast, and sharpness monitoring.
- Runtime and model-health evidence.

## Secure Hugging Face connection

Do not paste a token into the notebook source or into a chat message.

1. Create a Hugging Face user access token that can write repositories under `bhkr0901`.
2. In Colab, open **Secrets** from the left sidebar.
3. Add a secret named `HF_TOKEN` and enable notebook access.
4. The deployment cell reads the secret privately, verifies that it belongs to `bhkr0901` or an organization with that name, and never writes it into Drive, the generated ZIP, notebook output, or either repository.

When the artifact repository is public, the deployed Space needs no runtime token. When `HF_REPOSITORIES_PRIVATE = True`, the cell adds `HF_TOKEN` as a private Space secret so that the app can download the private artifacts.

## Recommended notebook execution

1. Open `DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb` in a fresh Colab runtime.
2. Select a GPU runtime.
3. Leave `PROFILE = "smoke"` for the first complete integration run.
4. Confirm the Hub controls:

```python
HF_USERNAME = "bhkr0901"
HF_SPACE_NAME = "defectspotter"
HF_MODEL_REPO_NAME = "defectspotter-artifacts"
HF_REPOSITORIES_PRIVATE = False
HF_SPACE_HARDWARE = "zero-a10g"
DEPLOY_TO_HUGGINGFACE = True
```

5. Use **Runtime → Run all** and approve Google Drive access.
6. Do not skip directly to packaging or deployment. The UI requires the trained artifacts and the saved EDA, evaluation, latency, threshold, and drift reports.
7. The packaging section runs the project tests, creates the local ZIP, saves it to Drive, and prepares attributed UI examples.
8. The Hugging Face section builds and uploads the backend and frontend bundles.
9. The deployment cell prints and displays the Space link after the Hub build starts. It also saves `huggingface_deployment.json` in the run's Drive folder.
10. After a successful smoke run, change to `PROFILE = "balanced"` and run the complete notebook again for the intended research result.

## Hugging Face Space eligibility

The notebook first requests `zero-a10g`, which is the current route for eligible free personal accounts in good standing. It then retries the account's default compute hardware. Hugging Face can reject Gradio Space creation when the account has neither an eligible ZeroGPU allowance nor a qualifying subscription or billing setup.

When creation is rejected, training is not lost. The generated Hub bundles remain in:

```text
MyDrive/DefectSpotter/runs/<RUN_ID>/huggingface_deploy/
```

After the account requirement is resolved, rerun only the Hugging Face deployment cell.

## Expected repositories and application

After a successful deployment:

```text
Model/evidence repository: bhkr0901/defectspotter-artifacts
Gradio Space:             bhkr0901/defectspotter
API endpoints:            /inspect_part
                          /inspect_batch
                          /health
```

The Space rebuilds automatically after uploaded source changes. The frontend retrieves the artifact snapshot indicated by the Space variable `MODEL_REPO_ID`.

## Standalone deployment outside Colab

The enhanced source contains `deploy_huggingface.py`. After a completed notebook run has created `DefectSpotter_Project`, deploy from a terminal without retraining:

```bash
export HF_TOKEN="your_write_scoped_token"
python deploy_huggingface.py \
  --project-root /absolute/path/to/DefectSpotter_Project \
  --username bhkr0901 \
  --space-name defectspotter \
  --model-repo-name defectspotter-artifacts \
  --space-hardware zero-a10g \
  --wait
```

The script first constructs auditable `model_repo` and `space_repo` folders, then uploads them.

## Troubleshooting

### No charts or empty dashboards

Run all analysis sections through packaging. The application expects `dataset_composition.csv`, `image_statistics.csv`, `metrics_by_category_method.csv`, `operating_metrics.csv`, `latency.csv`, `artifact_sizes.csv`, and `drift_stress_test.csv`.

### Image upload works but inspection fails

Confirm that the selected category has at least one complete method artifact and calibration file. Check the Runtime Health tab for loaded categories and methods.

### Webcam does not start

Open the Space over HTTPS, grant browser camera permission, and use a browser that supports webcam capture. The live view is a demonstration; it is not a PLC-synchronized production line loop.

### HTTP 401 or 403 during deployment

The token is missing, expired, lacks repository write permission, or does not belong to `bhkr0901`.

### HTTP 402 or a Space hardware error

The account is not currently eligible to create the requested compute Space. Resolve the account plan, billing, or ZeroGPU eligibility, then rerun only the deployment cell.

### Space build error

Open the Space build logs. Typical causes are a package-wheel mismatch, incomplete artifact upload, or a private model repository without a Space `HF_TOKEN` secret. The deployment record and bundle manifest identify the exact uploaded files.

### Space opens but shows the diagnostic page

The frontend could not resolve `MODEL_REPO_ID` or the artifact repository lacks the expected `artifacts/` and `reports/` folders. Verify the Space variable and backend repository contents.

## Real-world use boundary

MVTec AD is used only for research and teaching and is licensed CC BY-NC-SA 4.0 for non-commercial use. A production implementation must use plant-owned or otherwise appropriately licensed images, a fixed and qualified camera/lighting setup, time- and lot-separated validation, a challenge set containing critical defect modes, operator training, authenticated access, audit logging, and a documented response to CHECK decisions, drift, camera failure, and model unavailability.
