"""Frozen ImageNet backbones that turn an image into a grid of patch descriptors.

PatchCore-style recipe: take mid-level feature maps (stride 8 and 16), apply a
3x3 local average (each descriptor sees its neighbourhood), upsample the deeper
map to the shallower resolution and concatenate. Only the layers that are
actually used are kept, which is what makes the exported ONNX file small.

PyTorch is needed here (training/export only). Inference uses the ONNX file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import imagenet_input


@dataclass(frozen=True)
class BackboneSpec:
    name: str
    feature_dim: int
    stride: int
    description: str


BACKBONES: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec("resnet18", 128 + 256, 8, "ResNet-18 layer2 + layer3"),
    "resnet34": BackboneSpec("resnet34", 128 + 256, 8, "ResNet-34 layer2 + layer3"),
    "wide_resnet50_2": BackboneSpec("wide_resnet50_2", 512 + 1024, 8, "WideResNet-50-2 layer2 + layer3"),
    "mobilenet_v3_small": BackboneSpec("mobilenet_v3_small", 24 + 48 + 96, 8, "MobileNetV3-Small blocks 3/8/11"),
    "efficientnet_b0": BackboneSpec("efficientnet_b0", 40 + 80 + 112, 8, "EfficientNet-B0 stages 3/4/5"),
}


class PatchFeatureExtractor(nn.Module):
    """Image (B,3,H,W, ImageNet-normalised) -> patch features (B,C,H/8,W/8)."""

    def __init__(self, backbone: str = "resnet18", pretrained: bool = True, pool_size: int = 3) -> None:
        super().__init__()
        from torchvision import models

        if backbone not in BACKBONES:
            raise ValueError(f"Unknown backbone {backbone!r}; choose from {sorted(BACKBONES)}")
        self.backbone_name = backbone
        self.pool_size = int(pool_size)
        weights = "DEFAULT" if pretrained else None

        if backbone.startswith(("resnet", "wide_resnet")):
            net = getattr(models, backbone)(weights=weights)
            self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool, net.layer1)
            self.stages = nn.ModuleList([net.layer2, net.layer3])
            self.taps = [0, 1]
        elif backbone == "mobilenet_v3_small":
            feats = models.mobilenet_v3_small(weights=weights).features
            self.stem = nn.Sequential(*feats[:4])            # -> block 3 (stride 8)
            self.stages = nn.ModuleList([nn.Sequential(*feats[4:9]), nn.Sequential(*feats[9:12])])
            self.taps = [-1, 0, 1]
        elif backbone == "efficientnet_b0":
            feats = models.efficientnet_b0(weights=weights).features
            self.stem = nn.Sequential(*feats[:4])             # -> stage 3 (stride 8)
            self.stages = nn.ModuleList([nn.Sequential(feats[4]), nn.Sequential(feats[5])])
            self.taps = [-1, 0, 1]
        self.eval()
        for parameter in self.parameters():
            parameter.requires_grad_(False)

    @property
    def feature_dim(self) -> int:
        return BACKBONES[self.backbone_name].feature_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        maps: list[torch.Tensor] = [x] if -1 in self.taps else []
        for stage in self.stages:
            x = stage(x)
            maps.append(x)
        pad = self.pool_size // 2
        maps = [F.avg_pool2d(m, kernel_size=self.pool_size, stride=1, padding=pad, count_include_pad=False)
                for m in maps]
        size = maps[0].shape[-2:]
        maps = [maps[0]] + [F.interpolate(m, size=size, mode="bilinear", align_corners=False) for m in maps[1:]]
        return torch.cat(maps, dim=1)


preprocess = imagenet_input  # same preprocessing for training, export and ONNX inference


@torch.inference_mode()
def extract_features(model: PatchFeatureExtractor, items: Sequence, image_size: int, batch_size: int = 16,
                     device: str = "cpu", progress: Callable[[int, int], None] | None = None) -> np.ndarray:
    """Return patch features as (N, h, w, C) float16 to keep caches small."""
    model = model.to(device).eval()
    out: list[np.ndarray] = []
    for start in range(0, len(items), batch_size):
        batch = np.stack([preprocess(item, image_size) for item in items[start:start + batch_size]])
        features = model(torch.from_numpy(batch).to(device))
        out.append(features.permute(0, 2, 3, 1).float().cpu().numpy().astype(np.float16))
        if progress:
            progress(min(start + batch_size, len(items)), len(items))
    return np.concatenate(out, axis=0)


def export_onnx(model: PatchFeatureExtractor, path: str | Path, image_size: int, opset: int = 17) -> dict:
    """Export with a dynamic batch axis and verify against PyTorch with ONNX Runtime."""
    import onnxruntime as ort

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    example = torch.randn(2, 3, image_size, image_size)
    torch.onnx.export(
        model, (example,), str(path), input_names=["image"], output_names=["patch_features"],
        dynamic_axes={"image": {0: "batch"}, "patch_features": {0: "batch"}},
        opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.inference_mode():
        expected = model(example).numpy()
    got = session.run(None, {"image": example.numpy()})[0]
    max_error = float(np.max(np.abs(expected - got)))
    return {
        "path": path.name,
        "backbone": model.backbone_name,
        "image_size": image_size,
        "feature_dim": int(expected.shape[1]),
        "grid": [int(expected.shape[2]), int(expected.shape[3])],
        "size_mb": path.stat().st_size / 1024 ** 2,
        "max_abs_error_vs_torch": max_error,
        "verified": bool(max_error < 1e-3),
        "opset": opset,
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
