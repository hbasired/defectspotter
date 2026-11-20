"""Feature-space VAE: a convolutional VAE trained on CNN feature maps of good parts (the "better VAE").

Why: the pixel VAE (``vae.py``) must redraw every pixel through a 128-number bottleneck, so its reconstructions are
blurry everywhere and the reconstruction error barely separates defects from normal detail (screw AUROC ~0.5). This
model instead reconstructs what the frozen ResNet-18 *sees*: the 28x28 grid of 384-d patch descriptors (layer2 +
layer3, 3x3 averaged, 224 px input) - the same features PatchCore uses. Normal descriptors are reconstructed well;
a defect produces descriptors the decoder has never learned to produce, so the per-patch reconstruction error is a
sharp anomaly map (cf. deep-feature reconstruction methods for industrial anomaly detection).

It is still a variational autoencoder trained on good parts only (generative model, Week 8):
    encoder: 1x1 conv 384->256, 3x3/2 conv 256->128 (14x14), 3x3/2 conv 128->128 (7x7) -> mu, log-variance
             (a spatial latent of 32 x 7 x 7 = 1,568 numbers)
    decoder: mirrored with transposed convolutions back to 384 x 28 x 28
    loss:    MSE(reconstructed features, features) + beta * KL(q(z|x) || N(0, I))
The exported ONNX graph contains the per-channel feature normalisation and outputs the per-patch squared error
directly, so inference needs only ONNX Runtime + NumPy.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import cv2
import numpy as np

from .patchcore import topk_mean


@dataclass
class FeatureVAEConfig:
    image_size: int = 224          # input resolution of the shared feature extractor
    pool_size: int = 3             # 3x3-averaged descriptors (ONNX output "patch_features")
    feature_dim: int = 384
    hidden: tuple = (256, 128, 128)
    latent_channels: int = 32      # spatial latent: 32 x 7 x 7
    beta: float = 1e-3
    learning_rate: float = 2e-3
    epochs: int = 60
    batch_size: int = 16
    hflip: bool = True             # feature-map flip augmentation (off for screws: thread handedness)
    blur_sigma: float = 4.0
    topk_fraction: float = 0.01
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("history", None)
        payload["hidden"] = list(self.hidden)
        return payload


# ----------------------------------------------------------------- NumPy scoring
def score_error_map(error_map: np.ndarray, config: FeatureVAEConfig, output_size: int | None = None) -> tuple[float, np.ndarray]:
    """(h,w) per-patch squared error -> (image score, smoothed anomaly map at ``output_size``)."""
    size = int(output_size or config.image_size)
    amap = cv2.resize(np.asarray(error_map, np.float32), (size, size), interpolation=cv2.INTER_LINEAR)
    if config.blur_sigma > 0:
        amap = cv2.GaussianBlur(amap, (0, 0), config.blur_sigma * size / config.image_size)
    return float(topk_mean(amap, config.topk_fraction)), amap.astype(np.float32)


# ------------------------------------------------------------------- PyTorch side
def build_fvae(config: FeatureVAEConfig):
    import torch
    from torch import nn

    h1, h2, h3 = config.hidden

    class FeatureVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(config.feature_dim, h1, 1), nn.GroupNorm(8, h1), nn.SiLU(),
                nn.Conv2d(h1, h2, 3, 2, 1), nn.GroupNorm(8, h2), nn.SiLU(),
                nn.Conv2d(h2, h3, 3, 2, 1), nn.GroupNorm(8, h3), nn.SiLU())
            self.mu = nn.Conv2d(h3, config.latent_channels, 1)
            self.logvar = nn.Conv2d(h3, config.latent_channels, 1)
            self.decoder = nn.Sequential(
                nn.Conv2d(config.latent_channels, h3, 1), nn.SiLU(),
                nn.ConvTranspose2d(h3, h2, 4, 2, 1), nn.GroupNorm(8, h2), nn.SiLU(),
                nn.ConvTranspose2d(h2, h1, 4, 2, 1), nn.GroupNorm(8, h1), nn.SiLU(),
                nn.Conv2d(h1, config.feature_dim, 1))

        def encode(self, x):
            h = self.encoder(x)
            return self.mu(h), self.logvar(h).clamp(-8, 8)

        def forward(self, x):  # deterministic reconstruction (mean of q(z|x)) for scoring / export
            mu, _ = self.encode(x)
            return self.decoder(mu)

        def forward_train(self, x):
            mu, logvar = self.encode(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decoder(z), mu, logvar

    return FeatureVAE()


def train_fvae(train_features: np.ndarray, config: FeatureVAEConfig, seed: int = 42, device: str = "cpu",
               progress: Callable[[dict], None] | None = None, val_fraction: float = 0.1):
    """Train on (N, h, w, C) feature grids of GOOD parts. Returns (model, mean, std) with mean/std per channel."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    x = torch.from_numpy(np.asarray(train_features, np.float32)).permute(0, 3, 1, 2).contiguous()
    order = torch.randperm(len(x), generator=generator)
    n_val = max(2, int(round(len(x) * val_fraction)))
    val, fit = x[order[:n_val]], x[order[n_val:]]
    mean = fit.mean(dim=(0, 2, 3), keepdim=True)
    std = fit.std(dim=(0, 2, 3), keepdim=True) + 1e-6
    fit, val = (fit - mean) / std, (val - mean) / std
    model = build_fvae(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    steps = math.ceil(len(fit) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                                    total_steps=config.epochs * steps, pct_start=0.15)
    best, best_state, history = float("inf"), None, []
    val_d = val.to(device)
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        perm = torch.randperm(len(fit), generator=generator)
        total = 0.0
        for start in range(0, len(fit), config.batch_size):
            batch = fit[perm[start:start + config.batch_size]].to(device)
            if config.hflip and torch.rand(1, generator=generator).item() < 0.5:
                batch = batch.flip(-1)
            recon, mu, logvar = model.forward_train(batch)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = F.mse_loss(recon, batch) + config.beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            val_loss = float(F.mse_loss(model(val_d), val_d))
        row = {"epoch": epoch, "train_loss": total / len(fit), "val_loss": val_loss,
               "seconds": time.perf_counter() - started}
        history.append(row)
        if val_loss < best:
            best = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress:
            progress(row)
    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    config.history = history
    return model, mean.numpy().reshape(-1), std.numpy().reshape(-1)


def export_fvae_onnx(model, mean: np.ndarray, std: np.ndarray, config: FeatureVAEConfig, path: str | Path,
                     opset: int = 17) -> dict:
    """Export features (B,C,h,w) -> per-patch squared reconstruction error (B,h,w), normalisation included."""
    import onnxruntime as ort
    import torch
    from torch import nn

    class ErrorMap(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.vae = model
            self.register_buffer("mean", torch.from_numpy(np.asarray(mean, np.float32)).view(1, -1, 1, 1))
            self.register_buffer("std", torch.from_numpy(np.asarray(std, np.float32)).view(1, -1, 1, 1))

        def forward(self, features):
            z = (features - self.mean) / self.std
            return ((self.vae(z) - z) ** 2).mean(dim=1)

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ErrorMap().eval()
    grid = config.image_size // 8
    example = torch.randn(2, config.feature_dim, grid, grid)
    torch.onnx.export(wrapper, (example,), str(path), input_names=["features"], output_names=["error_map"],
                      dynamic_axes={"features": {0: "batch"}, "error_map": {0: "batch"}}, opset_version=opset,
                      do_constant_folding=True, dynamo=False)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = wrapper(example).numpy()
    got = session.run(None, {"features": example.numpy()})[0]
    error = float(np.max(np.abs(expected - got)))
    parameters = int(sum(p.numel() for p in model.parameters()))
    return {"path": path.name, "size_mb": path.stat().st_size / 1024 ** 2, "parameters": parameters,
            "max_abs_error_vs_torch": error, "verified": bool(error < 1e-3), "opset": opset}
