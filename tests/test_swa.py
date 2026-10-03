from __future__ import annotations

import time
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from rsna2026.config import load_config
from rsna2026.swa import configured_swa


def swa_settings(**overrides):
    return {
        "swa_enabled": True, "epochs": 20, "swa_start_epoch": 15,
        "swa_anneal_epochs": 2, "swa_backbone_learning_rate": 0.1,
        "swa_head_learning_rate": 0.2, **overrides,
    }


def test_swa_disabled_preserves_ema():
    assert configured_swa(torch.nn.Linear(1, 1), {"ema_decay": 0.999}) is None


@pytest.mark.parametrize("overrides, message", [
    ({"swa_start_epoch": 0}, "start epoch"),
    ({"swa_start_epoch": 20}, "start epoch"),
    ({"swa_anneal_epochs": 0}, "annealing"),
    ({"swa_anneal_epochs": 7}, "annealing"),
    ({"ema_decay": 0.999}, "EMA"),
    ({"swa_head_learning_rate": float("nan")}, "learning rates"),
    ({"swa_backbone_learning_rate": 0}, "learning rates"),
])
def test_swa_rejects_invalid_configuration(overrides, message):
    with pytest.raises(ValueError, match=message):
        configured_swa(torch.nn.Linear(1, 1), swa_settings(**overrides))


def test_swa_rejects_batchnorm_without_statistics_recalibration():
    with pytest.raises(ValueError, match="BatchNorm"):
        configured_swa(torch.nn.BatchNorm1d(2), swa_settings())


def test_swa_average_schedule_and_checkpoint_roundtrip(tmp_path):
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD([
        {"params": [model.weight], "lr": 0.4},
        {"params": [model.bias], "lr": 0.8},
    ])
    base = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=[0.4, 0.8], total_steps=20,
    )
    swa = configured_swa(model, swa_settings())
    for epoch in range(1, 21):
        scheduler = swa.begin_epoch(epoch, optimizer, base)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        model.weight.data.fill_(epoch)
        model.bias.data.fill_(epoch * 2)
        evaluated = swa.end_epoch(epoch, model)
        if epoch < 15:
            assert evaluated is model
            assert swa.average.n_averaged.item() == 0
        else:
            assert evaluated.weight.item() == pytest.approx((15 + epoch) / 2)
            assert evaluated.bias.item() == pytest.approx(15 + epoch)
    assert base.last_epoch == 14
    assert [group["lr"] for group in optimizer.param_groups] == pytest.approx([0.1, 0.2])
    from rsna2026.train import checkpoint_state_dict
    checkpoint = tmp_path / "swa.pt"
    torch.save(checkpoint_state_dict(evaluated), checkpoint)
    restored = torch.nn.Linear(1, 1)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    torch.testing.assert_close(restored(torch.ones(1, 1)), evaluated(torch.ones(1, 1)))
    assert swa.metadata(5)["selected_snapshots"] == 5
    assert swa.metadata(5)["completed_snapshots"] == 6


def test_swa_config_changes_only_averaging_settings():
    original = load_config("configs/dinov2-ranking.toml")
    averaged = load_config("configs/dinov2-ranking-swa.toml")
    averaged["training"] = {
        key: value for key, value in averaged["training"].items()
        if not key.startswith("swa_")
    }
    assert averaged == original


@pytest.mark.parametrize("best_epoch, completed_epochs", [(14, 20), (19, 20), (1, 1)])
def test_cv_selects_inner_best_and_uses_same_weights_for_outer(
    tmp_path, monkeypatch, best_epoch, completed_epochs,
):
    import numpy as np
    import pandas as pd
    from rsna2026 import train
    from rsna2026.constants import TARGETS

    config = load_config("configs/dinov2-ranking-swa.toml")
    config["training"]["num_workers"] = 0
    if completed_epochs == 1:
        config["training"]["max_runtime_hours"] = 0
    rows = [
        {"StudyInstanceUID": f"s{i}", **{t: float(i % 2) for t in TARGETS}}
        for i in range(6)
    ]
    table = pd.DataFrame(rows).set_index("StudyInstanceUID", drop=False)
    roles = pd.DataFrame({
        "StudyInstanceUID": list(table.index),
        "role": ["training"] * 2 + ["inner-validation"] * 2 + ["outer-holdout"] * 2,
        "is_gold": [True] * 6,
    })
    public = {f"s{i}": (0.8 if i % 2 else 0.2,) * 12 for i in range(6)}
    records = {uid: SimpleNamespace(study_uid=uid) for uid in table.index}

    class TinyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = torch.nn.Linear(1, 1)
            self.head = torch.nn.Linear(1, 12)

    model = TinyModel()
    monkeypatch.setattr(train, "build_configured_model", lambda *a: (model, "synthetic", 26, 26))
    monkeypatch.setattr(train, "make_loader", lambda *a: [a[3]])
    steps = []

    def fake_train(model, loader, optimizer, scheduler, *args, **kwargs):
        epoch = len(steps) + 1
        steps.append(scheduler is not None)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        for parameter in model.parameters():
            parameter.data.fill_(epoch)
        return 0.5

    evaluated_weights = []

    def fake_predict(model, loader, device):
        weight = model.backbone.weight.item()
        evaluated_weights.append(weight)
        labels = loader[0]
        predictions = np.full_like(labels, weight / 21)
        attention = np.zeros((len(labels), 12, 1), dtype=np.float32)
        return predictions, labels, attention, attention, attention

    scores = iter([0.9 if epoch == best_epoch else 0.7 for epoch in range(1, 21)])
    monkeypatch.setattr(train, "train_epoch", fake_train)
    monkeypatch.setattr(train, "predict", fake_predict)
    monkeypatch.setattr(train, "target_auc_scores", lambda *a: dict.fromkeys(TARGETS, next(scores)))
    summary, oof, stopped = train.train_cv_fold(
        config, 0, time.monotonic(), tmp_path, table, roles, public, records,
        torch.device("cpu"), train.DistributedContext(False),
    )
    expected = float(best_epoch) if best_epoch < 15 else 17.0
    assert summary["best_epoch"] == best_epoch
    assert summary["swa"]["selected_snapshots"] == (0 if best_epoch < 15 else 5)
    assert summary["swa"]["completed_snapshots"] == max(0, completed_epochs - 14)
    assert evaluated_weights[-1] == pytest.approx(expected)
    assert steps == ([True] * 14 + [False] * 6)[:completed_epochs]
    state = torch.load(tmp_path / "fold-0/best-model.pt", weights_only=True)
    assert state["backbone.weight"].item() == pytest.approx(expected)
    assert oof[f"prediction_{TARGETS[0]}"].iloc[0] == pytest.approx(expected / 21)
    assert stopped is (completed_epochs < 20)
