"""Optional epoch-wise SWA for models without batch normalization."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.optim.swa_utils import AveragedModel, SWALR


class StochasticWeightAverage:
    """Average late training epochs and switch to a constant learning rate."""

    def __init__(self, model: nn.Module, training: dict[str, object]) -> None:
        epochs = int(training["epochs"])
        self.start_epoch = int(training.get("swa_start_epoch", 15))
        self.anneal_epochs = int(training.get("swa_anneal_epochs", 2))
        if not 1 <= self.start_epoch < epochs:
            raise ValueError("SWA start epoch must allow at least two snapshots")
        if not 1 <= self.anneal_epochs <= epochs - self.start_epoch + 1:
            raise ValueError("SWA annealing epochs must fit the averaging period")
        if training.get("ema_decay") is not None:
            raise ValueError("SWA and EMA cannot be enabled together")
        self.learning_rates = [
            float(training.get("swa_backbone_learning_rate", 0.000004)),
            float(training.get("swa_head_learning_rate", 0.0005)),
        ]
        if any(not math.isfinite(lr) or lr <= 0 for lr in self.learning_rates):
            raise ValueError("SWA learning rates must be finite and positive")
        raw_model = getattr(model, "module", model)
        if any(isinstance(layer, nn.modules.batchnorm._BatchNorm)
               for layer in raw_model.modules()):
            raise ValueError(
                "SWA currently supports models without BatchNorm (such as DINOv2); "
                "BatchNorm models require training-only statistics recalibration"
            )
        self.average = AveragedModel(raw_model, use_buffers=False)
        self.average.requires_grad_(False)
        self.scheduler: SWALR | None = None

    def begin_epoch(
        self, epoch: int, optimizer: torch.optim.Optimizer,
        base_scheduler: torch.optim.lr_scheduler.LRScheduler,
    ) -> torch.optim.lr_scheduler.LRScheduler | None:
        """Return the per-step scheduler; SWA anneals once per epoch instead."""
        if epoch < self.start_epoch:
            return base_scheduler
        if self.scheduler is None:
            self.scheduler = SWALR(
                optimizer, swa_lr=self.learning_rates,
                anneal_epochs=self.anneal_epochs, anneal_strategy="cos",
            )
        return None

    def end_epoch(self, epoch: int, model: nn.Module) -> nn.Module:
        """Capture the completed epoch before validation, with equal weights."""
        if epoch < self.start_epoch:
            return model
        if self.scheduler is None:
            raise RuntimeError("SWA epoch must be started before averaging")
        self.average.update_parameters(getattr(model, "module", model))
        self.scheduler.step()
        return self.average.module

    def metadata(self, selected_snapshots: int) -> dict[str, object]:
        """Describe both the configured run and the selected checkpoint."""
        return {
            "start_epoch": self.start_epoch,
            "anneal_epochs": self.anneal_epochs,
            "backbone_learning_rate": self.learning_rates[0],
            "head_learning_rate": self.learning_rates[1],
            "completed_snapshots": int(self.average.n_averaged.item()),
            "selected_snapshots": selected_snapshots,
            "selected_weight_source": "swa" if selected_snapshots else "raw",
            "averaging": "equal-weight-per-completed-epoch",
        }


def configured_swa(
    model: nn.Module, training: dict[str, object],
) -> StochasticWeightAverage | None:
    """Leave existing experiments unchanged unless SWA is explicitly enabled."""
    if not training.get("swa_enabled", False):
        return None
    return StochasticWeightAverage(model, training)
