"""Optional validation-based stopping with a minimum improvement threshold."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EarlyStopping:
    """Track significant improvements independently for each training fold."""

    patience: int = 0
    min_delta: float = 0.0
    best_score: float = -math.inf
    stale_epochs: int = 0
    stopped: bool = False

    def __post_init__(self) -> None:
        if self.patience < 0:
            raise ValueError("Early stopping patience must be nonnegative")
        if not math.isfinite(self.min_delta) or self.min_delta < 0:
            raise ValueError("Early stopping minimum improvement must be finite and nonnegative")

    @classmethod
    def from_config(cls, training: dict[str, object]) -> EarlyStopping:
        return cls(
            patience=int(training.get("early_stopping_patience", 0)),
            min_delta=float(training.get("early_stopping_min_delta", 0.0)),
        )

    def update(self, score: float) -> bool:
        """Return whether the checkpoint improved; stop after patience misses."""
        if not math.isfinite(score):
            raise ValueError("Early stopping score must be finite")
        threshold = self.min_delta if self.patience else 0.0
        improved = score > self.best_score + threshold
        if improved:
            self.best_score = score
            self.stale_epochs = 0
        else:
            self.stale_epochs += 1
        self.stopped = self.patience > 0 and self.stale_epochs >= self.patience
        return improved

    def metadata(self) -> dict[str, object]:
        return {
            "enabled": self.patience > 0,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "stale_epochs": self.stale_epochs,
            "stopped": self.stopped,
        }
