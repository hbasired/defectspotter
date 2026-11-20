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

    def stage_maps(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.stem(x)
        maps: list[torch.Tensor] = [x] if -1 in self.taps else []
        for stage in self.stages:
            x = stage(x)
            maps.append(x)
        return maps

    @staticmethod
    def aggregate(maps: list[torch.Tensor], pool_size: int) -> torch.Tensor:
        """Local p x p average of every stage (PatchCore's neighbourhood aggregation), then concatenate.

        ``pool_size=1`` keeps the raw descriptors: fine texture detail that a 3x3 average smooths away
        (on carpet this alone lifts image AUROC from ~0.86 to ~0.93 with ResNet-18).
        """
        if pool_size > 1:
            pad = pool_size // 2
            maps = [F.avg_pool2d(m, kernel_size=pool_size, stride=1, padding=pad, count_include_pad=False)
                    for m in maps]
        size = maps[0].shape[-2:]
        maps = [maps[0]] + [F.interpolate(m, size=size, mode="bilinear", align_corners=False) for m in maps[1:]]
        return torch.cat(maps, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.aggregate(self.stage_maps(x), self.pool_size)

    def forward_pools(self, x: torch.Tensor, pools: Sequence[int]) -> list[torch.Tensor]:
        """One backbone pass, several aggregations (shared by training and the ONNX export)."""
        maps = self.stage_maps(x)
        return [self.aggregate(maps, int(p)) for p in pools]


class MultiPoolExport(nn.Module):
    """ONNX wrapper: one forward pass -> one output per pooling size (``patch_features_p<k>``)."""

    def __init__(self, extractor: PatchFeatureExtractor, pools: Sequence[int]) -> None:
        super().__init__()
        self.extractor = extractor
        self.pools = tuple(int(p) for p in pools)

    def forward(self, x: torch.Tensor):
        return tuple(self.extractor.forward_pools(x, self.pools))


EXPORT_POOLS = (3, 1)  # output 0 = PatchCore default (3x3), output 1 = raw descriptors


def output_name(pool_size: int) -> str:
    return "patch_features" if int(pool_size) == 3 else f"patch_features_p{int(pool_size)}"


preprocess = imagenet_input  # same preprocessing for training, export and ONNX inference


@torch.inference_mode()
def extract_features(model: PatchFeatureExtractor, items: Sequence, image_size: int, batch_size: int = 16,
                     device: str = "cpu", progress: Callable[[int, int], None] | None = None,
                     pools: Sequence[int] | None = None, augment: Callable | None = None):
    """Return patch features as (N, h, w, C) float16 to keep caches small.

    With ``pools`` a dict {pool_size: (N, h, w, C)} is returned from a single backbone pass.
    ``augment(item) -> list[RGB]`` expands every item into several views (e.g. rotations).
    """
    model = model.to(device).eval()
    pool_list = list(pools) if pools else [model.pool_size]
    out: dict[int, list[np.ndarray]] = {p: [] for p in pool_list}

    def batches():  # lazy: views (rotations, blends) are created batch by batch, never all at once
        buffer = []
        for item in items:
            for view in (augment(item) if augment is not None else [item]):
                buffer.append(preprocess(view, image_size))
                if len(buffer) == batch_size:
                    yield buffer
                    buffer = []
        if buffer:
            yield buffer

    done = 0
    for batch in batches():
        features = model.forward_pools(torch.from_numpy(np.stack(batch)).to(device), pool_list)
        for p, f in zip(pool_list, features):
            out[p].append(f.permute(0, 2, 3, 1).float().cpu().numpy().astype(np.float16))
        done += len(batch)
        if progress:
            progress(done, len(items))
    arrays = {p: np.concatenate(v, axis=0) for p, v in out.items()}
    return arrays if pools else arrays[pool_list[0]]


def export_onnx(model: PatchFeatureExtractor, path: str | Path, image_size: int, opset: int = 17,
                pools: Sequence[int] = EXPORT_POOLS, check_sizes: Sequence[int] = ()) -> dict:
    """Export with dynamic batch and spatial axes; verify against PyTorch with ONNX Runtime.

    The graph has one output per pooling size so every product can use its own aggregation
    (and its own input resolution) with a single shared 10-20 MB extractor.
    """
    import onnxruntime as ort

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model = model.cpu().eval()
    wrapper = MultiPoolExport(model, pools).eval()
    names = [output_name(p) for p in wrapper.pools]
    example = torch.randn(2, 3, image_size, image_size)
    dynamic = {"image": {0: "batch", 2: "height", 3: "width"}}
    dynamic.update({n: {0: "batch", 2: "grid_h", 3: "grid_w"} for n in names})
    torch.onnx.export(
        wrapper, (example,), str(path), input_names=["image"], output_names=names,
        dynamic_axes=dynamic, opset_version=opset, do_constant_folding=True, dynamo=False,
    )
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    max_error, grids = 0.0, {}
    for size in [image_size, *[s for s in check_sizes if s != image_size]]:
        x = torch.randn(1, 3, size, size)
        with torch.inference_mode():
            expected = [t.numpy() for t in wrapper(x)]
        got = session.run(None, {"image": x.numpy()})
        max_error = max(max_error, *[float(np.max(np.abs(e - g))) for e, g in zip(expected, got)])
        grids[int(size)] = [int(expected[0].shape[2]), int(expected[0].shape[3])]
    return {
        "path": path.name,
        "backbone": model.backbone_name,
        "image_size": image_size,
        "feature_dim": int(expected[0].shape[1]),
        "grid": grids[int(image_size)],
        "grids": {str(k): v for k, v in grids.items()},
        "outputs": {str(p): n for p, n in zip(wrapper.pools, names)},
        "dynamic_spatial": True,
        "size_mb": path.stat().st_size / 1024 ** 2,
        "max_abs_error_vs_torch": max_error,
        "verified": bool(max_error < 1e-3),
        "opset": opset,
    }


def count_parameters(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))
