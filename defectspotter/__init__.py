"""DefectSpotter: visual inspection without defect labels.

Good-only industrial anomaly detection with a PatchCore-style memory bank,
a convolutional VAE, an aligned mean-image baseline, synthetic-defect
threshold calibration, cost-aware PASS / CHECK / REJECT decisions and
camera-drift monitoring.

The package is split so that inference needs only NumPy, OpenCV and
ONNX Runtime. PyTorch is required only for training and ONNX export
(``features.py``, ``vae.py`` and ``pipeline.py``).
"""

__version__ = "2.0.0"
