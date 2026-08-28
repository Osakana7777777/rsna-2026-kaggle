from __future__ import annotations

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from rsna2026.dataset import KneeStudyDataset  # noqa: E402
from rsna2026.train import (  # noqa: E402
    predict,
    select_slice_window,
    slice_window_starts,
)


def test_window_starts_match_training_and_inference_policies() -> None:
    assert slice_window_starts(12, 3, 3) == (0, 3, 6, 9)
    assert slice_window_starts(12, 3, 1) == tuple(range(10))


def test_selects_the_same_ordered_window_from_every_slot() -> None:
    images = torch.arange(24).reshape(1, 24, 1, 1, 1).float()
    present = torch.ones((1, 24), dtype=torch.bool)
    present[0, 16] = False

    selected_images, selected_present = select_slice_window(
        images,
        present,
        num_slots=2,
        sampled_slices_per_slot=12,
        window_size=3,
        start=3,
    )

    assert selected_images[:, :, 0, 0, 0].tolist() == [[3, 4, 5, 15, 16, 17]]
    assert selected_present.tolist() == [[True, True, True, True, False, True]]


class WindowLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.starts: list[float] = []

    def forward(self, images, present):
        del present
        logits = images[:, 0, 0, 0, 0]
        self.starts.extend(logits.detach().cpu().tolist())
        return logits[:, None].expand(-1, 12)


def test_predict_averages_overlapping_window_probabilities() -> None:
    images = torch.arange(4).reshape(4, 1, 1, 1).float()
    present = torch.ones(4, dtype=torch.bool)
    dataset = [(images, present, torch.empty(0), torch.empty(0))]
    model = WindowLogitModel()

    predictions, labels = predict(
        model,
        DataLoader(dataset, batch_size=1),
        torch.device("cpu"),
        num_slots=1,
        sampled_slices_per_slot=4,
        window_size=2,
        inference_window_stride=1,
    )

    expected = torch.sigmoid(torch.tensor([0.0, 1.0, 2.0])).mean().item()
    assert model.starts == [0.0, 1.0, 2.0]
    assert labels is None
    assert predictions.shape == (1, 12)
    assert np.allclose(predictions, expected)
    assert not np.allclose(predictions, torch.sigmoid(torch.tensor(1.0)).item())


@pytest.mark.parametrize(
    ("sampled", "window", "stride"),
    [(0, 3, 1), (2, 3, 1), (12, 0, 1), (12, 3, 0)],
)
def test_rejects_invalid_window_policies(sampled, window, stride) -> None:
    with pytest.raises(ValueError):
        slice_window_starts(sampled, window, stride)


def test_dicom_cache_stores_normalized_images_as_uint8(tmp_path) -> None:
    cache_path = tmp_path / "study.npz"
    images = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1, 1)
    present = torch.tensor([True, True, True])

    KneeStudyDataset._write_cache(cache_path, images, present)

    with np.load(cache_path, allow_pickle=False) as cached:
        assert cached["images"].dtype == np.uint8
        assert cached["images"].reshape(-1).tolist() == [0, 128, 255]
        assert cached["present"].tolist() == [True, True, True]
