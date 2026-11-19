"""Convolutional VAE trained on good parts only; reconstruction error = anomaly signal.

The VAE learns to reproduce normal appearance through a low-dimensional
latent bottleneck. Defects are poorly reconstructed, so the per-pixel
difference (L1 + structural SSIM error) localises them. Training/export use
PyTorch; ``reconstruction_anomaly_map`` is NumPy so the ONNX runtime can use it.
"""
from __future__ import annotations

import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

import cv2
import numpy as np

from .data import resize_rgb, to_rgb_array
from .patchcore import topk_mean


@dataclass
class VAEConfig:
    image_size: int = 128
    latent_dim: int = 128
    base_channels: int = 32
    beta: float = 1e-3
    ssim_weight: float = 0.5
    learning_rate: float = 1e-3
    epochs: int = 60
    batch_size: int = 32
    hflip: bool = True
    rot90: bool = True
    blur_sigma: float = 2.0
    topk_fraction: float = 0.01
    history: list = field(default_factory=list)

    def to_dict(self) -> dict:
        payload = asdict(self)
        payload.pop("history", None)
        return payload


# ----------------------------------------------------------------- NumPy scoring
def load_vae_input(image, image_size: int) -> np.ndarray:
    """RGB image -> (3,S,S) float32 in [0,1]."""
    array = resize_rgb(to_rgb_array(image), image_size).astype(np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1))


def _ssim_map_gray(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    blur = lambda img: cv2.GaussianBlur(img, (0, 0), sigma)  # noqa: E731
    mu_a, mu_b = blur(a), blur(b)
    var_a = blur(a * a) - mu_a ** 2
    var_b = blur(b * b) - mu_b ** 2
    cov = blur(a * b) - mu_a * mu_b
    return ((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2))


def reconstruction_anomaly_map(x: np.ndarray, recon: np.ndarray, blur_sigma: float = 2.0) -> np.ndarray:
    """x, recon: (3,S,S) in [0,1] -> (S,S) anomaly map = 0.5*L1 + 0.5*(1-SSIM)."""
    l1 = np.abs(x - recon).mean(axis=0)
    gray_x = (0.299 * x[0] + 0.587 * x[1] + 0.114 * x[2]).astype(np.float32)
    gray_r = (0.299 * recon[0] + 0.587 * recon[1] + 0.114 * recon[2]).astype(np.float32)
    dssim = np.clip((1.0 - _ssim_map_gray(gray_x, gray_r)) / 2.0, 0, 1)
    anomaly = 0.5 * l1 + 0.5 * dssim
    if blur_sigma > 0:
        anomaly = cv2.GaussianBlur(anomaly.astype(np.float32), (0, 0), blur_sigma)
    return anomaly.astype(np.float32)


def score_reconstruction(x: np.ndarray, recon: np.ndarray, config: VAEConfig,
                         output_size: int | None = None) -> tuple[float, np.ndarray]:
    anomaly = reconstruction_anomaly_map(x, recon, config.blur_sigma)
    score = topk_mean(anomaly, config.topk_fraction)
    if output_size and output_size != anomaly.shape[0]:
        anomaly = cv2.resize(anomaly, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    return score, anomaly


# ------------------------------------------------------------------- PyTorch side
def _torch():
    import torch
    import torch.nn.functional as F
    from torch import nn

    return torch, F, nn


def build_vae(config: VAEConfig):
    torch, F, nn = _torch()

    class ConvVAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            b, s = config.base_channels, config.image_size
            if s % 16:
                raise ValueError("VAE image_size must be divisible by 16")

            def down(i, o):
                return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.SiLU(),
                                     nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())

            def up(i, o):
                return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.GroupNorm(8, o), nn.SiLU(),
                                     nn.Conv2d(o, o, 3, 1, 1), nn.GroupNorm(8, o), nn.SiLU())

            # A 1x1 "squeeze" before the fully-connected bottleneck keeps the model small
            # (~2.2M parameters, ~9 MB ONNX) while the latent stays 128-dimensional.
            squeeze = 32
            self.encoder = nn.Sequential(down(3, b), down(b, 2 * b), down(2 * b, 4 * b), down(4 * b, 4 * b),
                                         nn.Conv2d(4 * b, squeeze, 1))
            self.shape = (squeeze, s // 16, s // 16)
            flat = int(np.prod(self.shape))
            self.fc_mu = nn.Linear(flat, config.latent_dim)
            self.fc_logvar = nn.Linear(flat, config.latent_dim)
            self.fc_dec = nn.Linear(config.latent_dim, flat)
            self.decoder = nn.Sequential(nn.Conv2d(squeeze, 4 * b, 1), nn.SiLU(),
                                         up(4 * b, 4 * b), up(4 * b, 2 * b), up(2 * b, b),
                                         nn.ConvTranspose2d(b, b, 4, 2, 1), nn.SiLU(),
                                         nn.Conv2d(b, 3, 3, 1, 1), nn.Sigmoid())

        def encode(self, x):
            h = self.encoder(x).flatten(1)
            return self.fc_mu(h), self.fc_logvar(h).clamp(-12, 8)

        def decode(self, z):
            return self.decoder(self.fc_dec(z).view(z.shape[0], *self.shape))

        def forward(self, x):  # deterministic reconstruction used for scoring/export
            mu, _ = self.encode(x)
            return self.decode(mu)

        def forward_train(self, x):
            mu, logvar = self.encode(x)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decode(z), mu, logvar

    return ConvVAE()


def _ssim_torch(x, y, window):
    torch, F, _ = _torch()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    pad = window.shape[-1] // 2
    groups = x.shape[1]
    mu_x = F.conv2d(x, window, padding=pad, groups=groups)
    mu_y = F.conv2d(y, window, padding=pad, groups=groups)
    var_x = F.conv2d(x * x, window, padding=pad, groups=groups) - mu_x ** 2
    var_y = F.conv2d(y * y, window, padding=pad, groups=groups) - mu_y ** 2
    cov = F.conv2d(x * y, window, padding=pad, groups=groups) - mu_x * mu_y
    ssim = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2))
    return ssim.mean()


def _gaussian_window(channels: int = 3, size: int = 11, sigma: float = 1.5):
    torch, _, _ = _torch()
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    return (g[:, None] @ g[None, :])[None, None].repeat(channels, 1, 1, 1)


def _augment(batch, config: VAEConfig, generator):
    torch, _, _ = _torch()
    if config.hflip and torch.rand(1, generator=generator).item() < 0.5:
        batch = batch.flip(-1)
    if config.rot90:
        k = int(torch.randint(0, 4, (1,), generator=generator).item())
        batch = torch.rot90(batch, k, dims=(-2, -1))
    gain = 1.0 + (torch.rand(batch.shape[0], 1, 1, 1, generator=generator) - 0.5) * 0.08
    return (batch * gain).clamp(0, 1)


def train_vae(train_images: Sequence, val_images: Sequence, config: VAEConfig, seed: int = 42,
              progress: Callable[[dict], None] | None = None, device: str = "cpu"):
    """Train on good images only; keeps the checkpoint with the best validation loss."""
    torch, F, _ = _torch()
    torch.manual_seed(seed)
    generator = torch.Generator().manual_seed(seed)
    to_tensor = lambda items: torch.from_numpy(np.stack([load_vae_input(i, config.image_size) for i in items]))  # noqa: E731
    train_x, val_x = to_tensor(train_images), to_tensor(val_images)
    val_device = val_x.to(device)
    model = build_vae(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=1e-5)
    steps_per_epoch = math.ceil(len(train_x) / config.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.learning_rate,
                                                    total_steps=config.epochs * steps_per_epoch, pct_start=0.15)
    window = _gaussian_window().to(device)
    best_val, best_state, history = float("inf"), None, []
    for epoch in range(1, config.epochs + 1):
        started = time.perf_counter()
        model.train()
        order = torch.randperm(len(train_x), generator=generator)
        total = 0.0
        for start in range(0, len(train_x), config.batch_size):
            batch = _augment(train_x[order[start:start + config.batch_size]], config, generator).to(device)
            recon, mu, logvar = model.forward_train(batch)
            l1 = F.l1_loss(recon, batch)
            ssim_loss = 1.0 - _ssim_torch(recon, batch, window)
            kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
            loss = l1 + config.ssim_weight * ssim_loss + config.beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            total += float(loss.detach()) * len(batch)
        model.eval()
        with torch.inference_mode():
            recon = model(val_device)
            val_loss = float(F.l1_loss(recon, val_device) + config.ssim_weight * (1 - _ssim_torch(recon, val_device, window)))
        row = {"epoch": epoch, "train_loss": total / len(train_x), "val_loss": val_loss,
               "seconds": time.perf_counter() - started}
        history.append(row)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if progress:
            progress(row)
    model = model.cpu()
    model.load_state_dict(best_state)
    model.eval()
    config.history = history
    return model


def export_vae_onnx(model, config: VAEConfig, path: str | Path, opset: int = 17) -> dict:
    torch, _, _ = _torch()
    import onnxruntime as ort

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.rand(2, 3, config.image_size, config.image_size)
    torch.onnx.export(model, (example,), str(path), input_names=["image"], output_names=["reconstruction"],
                      dynamic_axes={"image": {0: "batch"}, "reconstruction": {0: "batch"}},
                      opset_version=opset, do_constant_folding=True, dynamo=False)
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(example).numpy()
    got = session.run(None, {"image": example.numpy()})[0]
    error = float(np.max(np.abs(expected - got)))
    return {"path": path.name, "size_mb": path.stat().st_size / 1024 ** 2, "max_abs_error_vs_torch": error,
            "verified": bool(error < 1e-3), "opset": opset}


def reconstruct_torch(model, images: Sequence, config: VAEConfig, batch_size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Return (inputs, reconstructions) as (N,3,S,S) float32 arrays."""
    torch, _, _ = _torch()
    inputs = np.stack([load_vae_input(i, config.image_size) for i in images])
    outs = []
    with torch.inference_mode():
        for start in range(0, len(inputs), batch_size):
            outs.append(model(torch.from_numpy(inputs[start:start + batch_size])).numpy())
    return inputs, np.concatenate(outs)
