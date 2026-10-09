"""Deep rungs (S4): model shapes and sizes, and the checkpoint/resume training loop."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from echofind.models.cnn import (  # noqa: E402
    ProfileCNN,
    SnippetCNN,
    n_params,
    profile_batch,
    snippet_batch,
)
from echofind.models.train import TrainConfig, fit, predict  # noqa: E402


def test_shapes_and_parameter_budgets():
    """One logit per candidate; R3a about 25k and R3b about 110k parameters (the plan's budget)."""
    assert ProfileCNN()(torch.rand(5, 128)).shape == (5,)
    assert SnippetCNN()(torch.rand(5, 64, 32)).shape == (5,)
    assert 20_000 < n_params(ProfileCNN()) < 30_000
    assert 90_000 < n_params(SnippetCNN()) < 130_000


def test_batches_scale_and_augment():
    rng = np.random.default_rng(0)
    snips = rng.integers(0, 256, (10, 64, 32), dtype=np.uint8)
    x = snippet_batch(snips, np.arange(10))
    assert x.dtype == torch.float32 and 0 <= float(x.min()) and float(x.max()) <= 1
    xa = snippet_batch(snips, np.arange(10), True, np.random.default_rng(1))
    assert xa.shape == x.shape
    profs = np.full((4, 128), 35.0, np.float16)
    assert torch.allclose(profile_batch(profs, np.arange(4)), torch.ones(4, 128))


def _toy():
    """Profiles with a bright bump at cell 40 for positives, noise for negatives."""
    rng = np.random.default_rng(0)
    p = rng.normal(0, 2, (600, 128)).astype(np.float16)
    y = (np.arange(600) % 3 == 0).astype(int)
    p[y == 1, 38:44] += 20
    return p, y


def test_fit_learns_toy_task_and_resumes(tmp_path):
    """The loop separates an obvious signal; a checkpoint at epoch 1 resumes to the same end."""
    p, y = _toy()

    def b(idx, aug, rng):
        return profile_batch(p, idx, aug, rng)

    tr, va = np.arange(0, 450), np.arange(450, 600)
    cfg = TrainConfig(epochs=3, batch=64, lr=3e-3, patience=5)
    torch.manual_seed(0)
    m = fit(ProfileCNN(), b, tr, y[tr], va, y[va], cfg, tmp_path / "a.pt", log=lambda s: None)
    s = predict(m, b, va)
    from sklearn.metrics import average_precision_score
    assert average_precision_score(y[va], s) > 0.95
    # resume: a run stopped after one epoch continues from its checkpoint
    torch.manual_seed(0)
    fit(ProfileCNN(), b, tr, y[tr], va, y[va], TrainConfig(epochs=1, batch=64, lr=3e-3),
        tmp_path / "b.pt", log=lambda s: None)
    msgs = []
    torch.manual_seed(0)
    fit(ProfileCNN(), b, tr, y[tr], va, y[va], cfg, tmp_path / "b.pt", log=msgs.append)
    assert msgs[0] == "resumed at epoch 1"
