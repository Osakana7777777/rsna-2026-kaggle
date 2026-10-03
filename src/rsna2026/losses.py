"""Confidence-weighted ranking losses for study predictions."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch.nn import functional
from torch.distributed.nn.functional import all_gather


def pairwise_logistic_ranking_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    weights: torch.Tensor,
    distributed: bool = False,
) -> torch.Tensor:
    """Average positive-negative logistic pairs across eligible targets.

    Probabilities above/below 0.5 define positive/negative studies. Pair weights
    are the product of cell confidence weights; undecided/missing cells are
    excluded. Differentiable gathering includes pairs across GPUs.
    """
    logits = logits.float()
    if distributed:
        logits = torch.cat(all_gather(logits), dim=0)
        gathered_labels = [torch.empty_like(labels) for _ in range(dist.get_world_size())]
        gathered_weights = [torch.empty_like(weights) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered_labels, labels)
        dist.all_gather(gathered_weights, weights)
        labels = torch.cat(gathered_labels, dim=0)
        weights = torch.cat(gathered_weights, dim=0)
    valid = torch.isfinite(labels) & torch.isfinite(weights) & (weights > 0)
    weights = torch.where(valid, weights.float(), 0.0)
    pair_mask = (
        (valid & (labels > 0.5))[:, None, :]
        & (valid & (labels < 0.5))[None, :, :]
    )
    pair_weights = weights[:, None, :] * weights[None, :, :] * pair_mask
    differences = logits[:, None, :] - logits[None, :, :]
    totals = pair_weights.sum(dim=(0, 1))
    per_target = (
        functional.softplus(-differences) * pair_weights
    ).sum(dim=(0, 1)) / totals.clamp_min(torch.finfo(torch.float32).tiny)
    eligible = totals > 0
    return (per_target * eligible).sum() / eligible.sum().clamp_min(1)
