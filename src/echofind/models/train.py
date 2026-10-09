"""CPU training loop for the deep rungs: weighted BCE, per-epoch checkpoints with resume,
early stopping on held-out sessions (candidate-level average precision).

A run killed mid-way (reclaimed VM, 30-minute limit) restarts from its last finished epoch:
the checkpoint holds the model, optimiser, RNG state and the best weights so far.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score
from torch import nn

Batcher = Callable[[np.ndarray, bool, np.random.Generator], torch.Tensor]


@dataclass
class TrainConfig:
    epochs: int = 12
    batch: int = 256
    lr: float = 2e-3
    weight_decay: float = 1e-4
    patience: int = 4
    seed: int = 0
    max_minutes: float = 28.0


def predict(model: nn.Module, batcher: Batcher, idx: np.ndarray, batch: int = 1024) -> np.ndarray:
    model.eval()
    out = np.empty(len(idx), np.float32)
    with torch.no_grad():
        for s in range(0, len(idx), batch):
            out[s: s + batch] = model(batcher(idx[s: s + batch], False, None)).numpy()
    return out


def fit(model: nn.Module, batcher: Batcher, tr_idx: np.ndarray, y_tr: np.ndarray,
        va_idx: np.ndarray, y_va: np.ndarray, cfg: TrainConfig, ckpt: Path,
        log: Callable[[str], None] = print) -> nn.Module:
    """Train with pos_weight = n_neg / n_pos; keep the epoch with the best validation AP."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    pos_w = torch.tensor((y_tr == 0).sum() / max((y_tr == 1).sum(), 1), dtype=torch.float32)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.epochs)
    start, best_ap, best_state, bad = 0, -1.0, None, 0
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    if ckpt.exists():
        st = torch.load(ckpt, weights_only=False)
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        rng.bit_generator.state = st["rng"]
        start, best_ap = st["epoch"] + 1, st["best_ap"]
        best_state, bad = st["best_state"], st["bad"]
        log(f"resumed at epoch {start}")
    t0 = time.time()
    yt = torch.from_numpy(y_tr.astype(np.float32))
    for ep in range(start, cfg.epochs):
        model.train()
        order = rng.permutation(len(tr_idx))
        tot = 0.0
        for s in range(0, len(order), cfg.batch):
            b = order[s: s + cfg.batch]
            x = batcher(tr_idx[b], True, rng)
            loss = lossf(model(x), yt[b])
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss.detach()) * len(b)
        sched.step()
        ap = float(average_precision_score(y_va, predict(model, batcher, va_idx)))
        if ap > best_ap:
            best_ap, best_state, bad = ap, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "rng": rng.bit_generator.state, "epoch": ep,
                    "best_ap": best_ap, "best_state": best_state, "bad": bad}, ckpt)
        log(f"epoch {ep}: loss {tot / len(order):.4f} val AP {ap:.4f} (best {best_ap:.4f}) "
            f"{(time.time() - t0) / 60:.1f} min")
        if bad >= cfg.patience or (time.time() - t0) / 60 > cfg.max_minutes:
            break
    model.load_state_dict(best_state)
    return model
