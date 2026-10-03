from __future__ import annotations

import math
import sys

import pytest

torch = pytest.importorskip("torch")

from rsna2026.config import load_config
from rsna2026.losses import pairwise_logistic_ranking_loss


def test_ranking_value_and_gradient_direction():
    logits = torch.tensor([[0.0], [0.0]], requires_grad=True)
    labels = torch.tensor([[1.0], [0.0]])
    loss = pairwise_logistic_ranking_loss(logits, labels, torch.ones_like(labels))
    assert loss.item() == pytest.approx(math.log(2))
    loss.backward()
    assert logits.grad[0].item() < 0
    assert logits.grad[1].item() > 0
    assert pairwise_logistic_ranking_loss(
        torch.tensor([[2.0], [-2.0]]), labels, torch.ones_like(labels)
    ) < loss


def test_confidence_weights_and_masking():
    logits = torch.tensor([[2.0], [-1.0], [0.0], [99.0], [99.0]])
    labels = torch.tensor([[1.0], [0.0], [0.0], [0.5], [float("nan")]])
    weights = torch.tensor([[3.0], [1.0], [0.5], [1.0], [1.0]])
    expected = (torch.nn.functional.softplus(torch.tensor(-3.0)) * 3
                + torch.nn.functional.softplus(torch.tensor(-2.0)) * 1.5) / 4.5
    assert pairwise_logistic_ranking_loss(logits, labels, weights).item() == pytest.approx(
        expected.item()
    )


def test_no_pairs_has_zero_loss_and_finite_gradients():
    logits = torch.zeros(3, 2, requires_grad=True)
    labels = torch.ones_like(logits)
    loss = pairwise_logistic_ranking_loss(logits, labels, torch.ones_like(logits))
    loss.backward()
    assert loss.item() == 0
    assert torch.equal(logits.grad, torch.zeros_like(logits))


def test_ranking_configuration():
    config = load_config("configs/dinov2-ranking.toml")
    training = config["training"]
    assert config["model"]["name"] == "dinov2-small"
    assert training["distributed"] is True
    assert training["distributed_strategy"] == "data-parallel"
    assert training["expected_gpu_count"] == 2
    assert training["batch_size_per_device"] == 4
    assert training["gradient_accumulation_steps"] == 1
    assert training["pairwise_ranking_loss_weight"] == 0.05
    assert config["cross_validation"]["folds"] == 4
    assert "run_folds" not in config["cross_validation"]
    assert config["cross_validation"]["gold_loss_weight"] == 3
    assert config["pseudo_labels"]["confidence_weight_method"] == "linear-distance"


def _distributed_gradient_worker(rank, rendezvous, result_dir):
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        model = torch.nn.Linear(1, 1, bias=False)
        model.weight.data.fill_(0.5)
        model = DistributedDataParallel(model)
        inputs = torch.tensor([[1.0], [2.0]]) if rank == 0 else torch.tensor([[-1.0], [-2.0]])
        labels = torch.ones_like(inputs) if rank == 0 else torch.zeros_like(inputs)
        loss = pairwise_logistic_ranking_loss(
            model(inputs), labels, torch.ones_like(inputs), distributed=True
        )
        loss.backward()
        torch.save((loss.detach(), model.module.weight.grad), result_dir / f"{rank}.pt")
    finally:
        dist.destroy_process_group()


def test_cross_rank_pairs_match_single_process_gradients(tmp_path, monkeypatch):
    if sys.platform == "darwin":
        monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo0")
    torch.multiprocessing.spawn(
        _distributed_gradient_worker,
        args=(f"file://{tmp_path / 'rendezvous'}", tmp_path), nprocs=2, join=True,
    )
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(0.5)
    inputs = torch.tensor([[1.0], [2.0], [-1.0], [-2.0]])
    labels = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    loss = pairwise_logistic_ranking_loss(model(inputs), labels, torch.ones_like(labels))
    loss.backward()
    for rank in range(2):
        distributed_loss, gradient = torch.load(tmp_path / f"{rank}.pt", weights_only=True)
        torch.testing.assert_close(distributed_loss, loss.detach())
        torch.testing.assert_close(gradient, model.weight.grad)
