# DefectSpotter — deployment guide (local · Colab · Hugging Face · Claude)

DefectSpotter ships as **one self-contained app**: `app.py` + the `defectspotter/` inference package + `ui/` + the trained
artifacts (`artifacts/`, ≈80 MB) + the evidence the dashboard plots (`reports/`, `examples/`). Inference needs only
NumPy, OpenCV and **ONNX Runtime** — no PyTorch — so it runs on any laptop CPU and on the CPU of a free Hugging Face Space.

## 1 · Run locally (after `git clone`)

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt    # runtime only: numpy, opencv, onnxruntime, gradio, plotly, ...
python app.py                      # open http://127.0.0.1:7860
```

Options: `python app.py --share` (temporary public `gradio.live` link), `--port 8080`, `--host 0.0.0.0`.
Python 3.10–3.13 are supported.

### Re-train everything from scratch (optional)

```bash
pip install -r requirements-train.txt                 # adds torch / torchvision / onnx
python scripts/download_data.py --dest data/mvtec_ad   # 1.3 GB, MVTec AD (CC BY-NC-SA 4.0)
python scripts/train.py --data data/mvtec_ad           # ~1 h on a 4-core CPU, ~15 min on a GPU
pip install -r requirements-diffusion.txt              # optional GenAI synthetic defects
python scripts/generate_diffusion_defects.py --data data/mvtec_ad
python scripts/train.py --data data/mvtec_ad --stages severity diffusion calibrate evaluate drift summary
```

## 2 · Google Colab

Open `DefectSpotter_End_to_End_Colab_HUGGINGFACE.ipynb` in Colab (GPU runtime recommended) and run all cells. It clones
this repository, runs every stage with inline charts (EDA → synthetic & diffusion defects → PatchCore / VAE / baseline →
calibration → evaluation → latency → drift → few-shot), runs the tests, launches the same Gradio dashboard with a public
link and (optionally) deploys to Hugging Face using the Colab secret `HF_TOKEN`.

## 3 · Hugging Face Spaces — what is free in 2026

From the official Spaces documentation (checked November 2026):

| Space type | Free personal account | Notes |
|---|---|---|
| **Gradio on ZeroGPU** | ✅ up to **2 Spaces** (verified e-mail, account older than 30 days) | the free route for a Python/Gradio app |
| Gradio / Docker on CPU-basic | ❌ needs PRO ($9/month) or a Team plan | hardware itself is free, creation is gated |
| Static (HTML/JS) | ✅ free for everyone | no Python backend |
| Streamlit | via the Docker SDK only (no longer a built-in SDK) | hence Gradio for this project |

DefectSpotter is **ZeroGPU-compatible**: `app.py` imports `spaces` first and registers one small `@spaces.GPU`
self-test (ZeroGPU refuses to start without one). All inspection runs on the container's CPU with ONNX Runtime, so
visitors never consume the daily GPU quota (2 min unauthenticated / 5 min free / 40 min PRO).

### Deploy from a terminal

```bash
export HF_TOKEN=hf_xxx          # a WRITE token from https://huggingface.co/settings/tokens - never commit it
python scripts/deploy_hf.py --space bhkr0901/defectspotter            # ZeroGPU (default)
python scripts/deploy_hf.py --space bhkr0901/defectspotter --dry-run  # just build the bundle
```

The script stages `app.py`, `defectspotter/`, `ui/`, `artifacts/`, `reports/`, `examples/`, writes the Space card
(`sdk: gradio`, pinned Gradio version, Python 3.12) and a Space-specific `requirements.txt`, creates the Space with
`zero-a10g` hardware (falling back to the account default with a clear message if the account is not eligible) and uploads.

* Space page: `https://huggingface.co/spaces/bhkr0901/defectspotter`
* Direct app: `https://bhkr0901-defectspotter.hf.space`
* API: `/inspect_part`, `/inspect_batch`, `/health` (see "Use via API" in the app footer)

```python
from gradio_client import Client, handle_file
client = Client("bhkr0901/defectspotter")
record, overlay = client.predict(handle_file("part.png"), "bottle", "patchcore", None, None, api_name="/inspect_part")
```

### Optional: separate model repository

If you prefer to version models separately, upload `artifacts/`, `reports/` and `examples/` to a model repo
(e.g. `bhkr0901/defectspotter-artifacts`), remove them from the Space and set the Space variable
`MODEL_REPO_ID=bhkr0901/defectspotter-artifacts`; `app.py` downloads them at start-up (add a read token as the Space secret
`HF_TOKEN` if the model repo is private).

### Troubleshooting

| Symptom | Cause / fix |
|---|---|
| 401 / 403 on deploy | token missing, expired or read-only |
| 402 / hardware rejected | account not eligible for ZeroGPU yet (age < 30 days or unverified e-mail): wait, or use PRO + `--hardware cpu-basic` |
| "No @spaces.GPU function detected" | make sure `app.py` is the Space entry point (it registers `gpu_self_test`) |
| Build error | open the Space build logs; the Space requirements are written by `deploy_hf.py` |
| Webcam does not start | the page must be served over HTTPS (Spaces are) and the browser must allow camera access |

## 4 · Can Claude deploy it? Can the backend run "in Claude"?

Verified in a Claude Code cloud session (November 2026):

* **Hugging Face Space — yes, and this is the frontend URL.** You save a write token once through the secure credential
  form (never in chat); Claude runs `scripts/deploy_hf.py`, waits for the build, smoke-tests `/inspect_part` and `/health`
  through the API and returns `https://huggingface.co/spaces/bhkr0901/defectspotter`. Ask Claude at any time to redeploy
  (after retraining or a code change), check the Space status or run test inspections against it.
* **Backend inside the Claude session — only privately.** Claude can start `python app.py` inside its sandbox and use it
  (API calls, automated tests, screenshots), but the sandbox accepts no inbound connections and blocks tunnels such as
  `--share` (gradio.live), so it cannot give you a public URL to that process, and the process stops when the session's
  container is recycled. The Space is the always-on backend + frontend.
* **Claude Artifact** — Claude can publish a static web page on claude.ai (private by default, shareable by link). It can
  host the project presentation/report; it cannot run the Python backend, so live inspection stays on the Space (or would
  have to be ported to ONNX Runtime Web in the browser).

## 5 · Production boundary

MVTec AD is CC BY-NC-SA 4.0 (non-commercial). A real deployment needs plant-owned images, a qualified fixed camera and
lighting, lot/time-separated validation including every critical defect mode, authenticated access, audit logging,
retention rules, a documented response to CHECK decisions, drift alarms, camera failure and model unavailability.
