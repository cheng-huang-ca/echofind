"""Deep ladder rungs R3 and R4 (docs/PLAN.md, W3; session S4), CPU-sized.

Inputs come from scripts/s4_snippets.py, cut from the same range-normalised grid as R2's
features: a 64 x 32 snippet (2.56 m of range by 32 beam cells, peak at row 24 so the shadow
behind the target is inside) and a 128-cell range profile through the peak beam.

- R3a ProfileCNN: 1-D CNN on the range profile (about 25k parameters). This is the input a
  single-beam handheld sonar has.
- R3b SnippetCNN: 2-D CNN on the snippet (about 110k parameters).
- R4a ResNet-50, ImageNet weights, frozen: 2048-d pooled features cached once, then a
  logistic-regression head (the Nga et al. 2024 backbone, used the CPU-friendly way).
- R4b MobileNetV3-Small, ImageNet weights, fine-tuned end to end (the edge-sized option).

Pretrained backbones take the grey snippet resized to 128 x 64 and repeated over 3 channels,
with ImageNet normalisation.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

IMAGENET_MEAN, IMAGENET_STD = 0.449, 0.226  # grey equivalents of the RGB statistics


def n_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


class ProfileCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o, k):
            return nn.Sequential(nn.Conv1d(i, o, k, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())

        self.f = nn.Sequential(blk(1, 16, 7), nn.MaxPool1d(2), blk(16, 32, 5), nn.MaxPool1d(2),
                               blk(32, 64, 5), nn.MaxPool1d(2), blk(64, 64, 3))
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(128, 1))

    def forward(self, x):            # x: (B, 128) dB-scaled profile
        h = self.f(x.unsqueeze(1))
        h = torch.cat([h.mean(-1), h.amax(-1)], 1)
        return self.head(h).squeeze(1)


class SnippetCNN(nn.Module):
    def __init__(self):
        super().__init__()

        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU())

        self.f = nn.Sequential(blk(1, 16), blk(16, 16), nn.MaxPool2d(2),
                               blk(16, 32), blk(32, 32), nn.MaxPool2d(2),
                               blk(32, 64), blk(64, 64), nn.MaxPool2d(2), blk(64, 64))
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):            # x: (B, 64, 32)
        h = self.f(x.unsqueeze(1))
        h = torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], 1)
        return self.head(h).squeeze(1)


def _to_rgb(x: torch.Tensor) -> torch.Tensor:
    """(B, 64, 32) in [0, 1] -> (B, 3, 128, 64), ImageNet-normalised."""
    x = F.interpolate(x.unsqueeze(1), size=(128, 64), mode="bilinear", align_corners=False)
    return ((x - IMAGENET_MEAN) / IMAGENET_STD).expand(-1, 3, -1, -1)


class MobileNetSmall(nn.Module):
    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

        w = MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        m = mobilenet_v3_small(weights=w)
        self.features = m.features
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(576 * 2, 1))

    def forward(self, x):
        h = self.features(_to_rgb(x))
        h = torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], 1)
        return self.head(h).squeeze(1)


class ResNet50Features(nn.Module):
    """Frozen ImageNet ResNet-50 to 2048-d average-pooled features."""

    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet50_Weights, resnet50

        m = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        m.fc = nn.Identity()
        self.m = m.eval()
        for p in self.m.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, x):
        return self.m(_to_rgb(x))


# ------------------------------------------------------------------ data helpers
def snippet_batch(snips: np.ndarray, idx: np.ndarray, augment: bool = False,
                  rng: np.random.Generator | None = None) -> torch.Tensor:
    """uint8 snippets -> float [0, 1]; augmentation flips the beam axis (a body seen from the
    other side) and shifts by up to 2 cells in range and beam."""
    x = snips[idx]
    x = torch.from_numpy(np.ascontiguousarray(x)).float() / 255.0
    if augment:
        rng = rng or np.random.default_rng()
        flip = torch.from_numpy(rng.random(len(x)) < 0.5)
        x[flip] = x[flip].flip(-1)
        dy, dx = (int(v) for v in rng.integers(-2, 3, 2))
        x = torch.roll(x, (dy, dx), (1, 2))
    return x


def profile_batch(profs: np.ndarray, idx: np.ndarray, augment: bool = False,
                  rng: np.random.Generator | None = None) -> torch.Tensor:
    """float16 dB profiles -> float, scaled to about [0, 1] ((dB + 5) / 40)."""
    x = torch.from_numpy(np.ascontiguousarray(profs[idx]).astype(np.float32))
    x = (x + 5.0) / 40.0
    if augment:
        rng = rng or np.random.default_rng()
        x = torch.roll(x, int(rng.integers(-2, 3)), 1)
    return x
