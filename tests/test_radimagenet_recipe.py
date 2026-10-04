from __future__ import annotations

import json
import math
import time
from copy import deepcopy
from types import SimpleNamespace

import pytest

from rsna2026.config import load_config
from rsna2026.early_stopping import EarlyStopping


def test_public_recipe_preserves_pixels_backbone_and_head() -> None:
    base = load_config("configs/radimagenet-resnet50-224.toml")
    recipe = load_config("configs/radimagenet-resnet50-public-recipe.toml")
    assert recipe["model"] == base["model"]
    assert recipe["model"]["unfreeze_last"] == 0
    for key in ("data", "image_cache", "augmentation", "cross_validation"):
        assert recipe[key] == base[key]
    assert recipe["training"]["epochs"] == 24
    assert recipe["training"]["head_learning_rate"] == 2e-4
    assert recipe["training"]["scheduler"] == "constant"
    assert recipe["training"]["early_stopping_patience"] == 5
    assert recipe["training"]["early_stopping_min_delta"] == 2e-4
    assert "ema_decay" not in recipe["training"]
    assert (recipe["training"]["batch_size_per_device"]
            * recipe["training"]["gradient_accumulation_steps"]) == 48
    labels = deepcopy(recipe["pseudo_labels"])
    assert labels.pop("use_explicit_supervision") is True
    assert labels == base["pseudo_labels"]
    metadata = json.loads(open("kaggle/radimagenet-kernel-metadata.json").read())
    assert recipe["pseudo_labels"]["dataset"] in metadata["dataset_sources"]
    assert "osakanateishoku/radimagenet-resnet50" in metadata["dataset_sources"]
    assert metadata["id"].endswith("rsna-2026-radimagenet-resnet50")


def test_stopping_uses_five_consecutive_significant_improvement_misses() -> None:
    stopper = EarlyStopping(5, 2e-4)
    assert stopper.update(0.8)
    for score in [0.8001, 0.8 + 2e-4, 0.799, 0.80015]:
        assert not stopper.update(score)
        assert not stopper.stopped
    assert not stopper.update(0.8001)
    assert stopper.stopped
    assert stopper.best_score == 0.8


def test_significant_improvement_resets_patience_and_folds_are_independent() -> None:
    stopper = EarlyStopping(5, 2e-4)
    stopper.update(0.8)
    stopper.update(0.8)
    assert stopper.update(0.801)
    assert stopper.stale_epochs == 0
    assert EarlyStopping(5, 2e-4).update(0.6)
    disabled = EarlyStopping()
    disabled.update(0.8)
    assert disabled.update(0.80001)
    for _ in range(30):
        disabled.update(0.7)
    assert not disabled.stopped


@pytest.mark.parametrize("patience,delta", [(-1, 0), (5, -1), (5, math.nan)])
def test_stopping_rejects_invalid_settings(patience, delta) -> None:
    with pytest.raises(ValueError):
        EarlyStopping(patience, delta)


def test_fixed_scheduler_and_explicit_supervision() -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from rsna2026 import train
    from rsna2026.constants import TARGETS

    parameter = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.AdamW([parameter], lr=2e-4)
    assert train.configured_scheduler(optimizer, {"scheduler": "constant"}, 4) is None
    for _ in range(10):
        parameter.grad = torch.ones_like(parameter)
        optimizer.step()
    assert optimizer.param_groups[0]["lr"] == 2e-4
    with pytest.raises(ValueError, match="Unsupported"):
        train.configured_scheduler(optimizer, {"scheduler": "invalid"}, 4)
    with pytest.raises(ValueError, match="SWA"):
        train.configured_scheduler(optimizer, {"scheduler": "constant", "swa_enabled": True}, 4)
    table = pd.DataFrame([
        {"StudyInstanceUID": uid, **{t: value for t in TARGETS}}
        for uid, value in [("gold", 1.), ("keep", np.nan), ("drop", np.nan)]
    ])
    teacher = {uid: (.8,) * 12 for uid in ["gold", "keep", "drop"]}
    confidence = {uid: (.4,) * 12 for uid in teacher}
    masks = {"gold": 0, "keep": 1, "drop": 0}
    labels, weights = train.supervision_arrays(table, teacher, 14., True,
                                              "legacy-distance", confidence, masks)
    np.testing.assert_array_equal(weights[0], [14.] * 12)
    np.testing.assert_allclose(weights[1], [.4] * 12)
    np.testing.assert_array_equal(weights[2], [0.] * 12)
    np.testing.assert_allclose(labels[1:], [[.8] * 12] * 2)


def test_fold_stops_and_saves_the_best_epoch_without_loading_images(tmp_path, monkeypatch) -> None:
    torch = pytest.importorskip("torch")
    np = pytest.importorskip("numpy")
    pd = pytest.importorskip("pandas")
    from rsna2026 import train
    from rsna2026.constants import TARGETS

    config = load_config("configs/radimagenet-resnet50-public-recipe.toml")
    model = torch.nn.Module()
    model.backbone = torch.nn.Linear(1, 1)
    model.backbone.requires_grad_(False)
    model.head = torch.nn.Linear(1, 1)
    rows = [{"StudyInstanceUID": f"study-{i}", **{t: float(i % 2) for t in TARGETS}}
            for i in range(6)]
    indexed = pd.DataFrame(rows).set_index("StudyInstanceUID", drop=False)
    roles = pd.DataFrame({"StudyInstanceUID": list(indexed.index),
                          "role": ["training"] * 2 + ["inner-validation"] * 2 + ["outer-holdout"] * 2,
                          "is_gold": [True] * 6})
    records = {uid: SimpleNamespace(study_uid=uid) for uid in indexed.index}
    public = {uid: (.8,) * 12 for uid in indexed.index}
    monkeypatch.setattr(train, "make_loader", lambda records, size, fov, labels, weights, *args: (labels, weights))
    monkeypatch.setattr(train, "build_configured_model", lambda *args: (model, tmp_path, 4, 2))
    epochs = []

    def fake_epoch(*args, **kwargs):
        epochs.append(1)
        with torch.no_grad():
            model.head.weight.fill_(len(epochs))
        return .5

    monkeypatch.setattr(train, "train_epoch", fake_epoch)
    monkeypatch.setattr(train, "target_auc_scores", lambda *args: {t: .8 for t in TARGETS})
    monkeypatch.setattr(train, "predict", lambda model, loader, device: (
        np.full_like(loader[0], .8), loader[0], np.zeros((2, 12, 6, 1, 10)),
        np.zeros((2, 12, 6, 1)), np.zeros((2, 12, 6)),
    ))
    summary, oof, time_limit = train.train_cv_fold(
        config, 0, time.monotonic(), tmp_path, indexed, roles, public,
        records, torch.device("cpu"), train.DistributedContext(False),
    )
    assert len(epochs) == 6
    assert summary["best_epoch"] == 1
    assert summary["completed_epochs"] == 6
    assert summary["early_stopping"]["stopped"] is True
    assert time_limit is False
    saved = torch.load(tmp_path / "fold-0/best-model.pt", weights_only=True)
    assert saved["head.weight"].item() == 1.
    assert len(oof) == 2
