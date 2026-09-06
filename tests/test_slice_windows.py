from __future__ import annotations

import pytest


np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from rsna2026.dataset import KneeStudyDataset  # noqa: E402
from rsna2026.train import predict  # noqa: E402


class WindowLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.starts: list[float] = []

    def forward(self, images, present, positions, return_attention=False):
        del positions
        logits = images[:, 0, 0, 0, 0, 0]
        self.starts.extend(logits.detach().cpu().tolist())
        logits = logits[:, None].expand(-1, 12)
        if not return_attention:
            return logits
        batch, slots, windows = present.shape
        window_attention = images.new_zeros((batch, 12, slots, windows))
        window_attention[..., 0] = 1.0
        slot_attention = images.new_ones((batch, 12, slots)) / slots
        return logits, window_attention, slot_attention


def test_predict_uses_all_dense_windows_in_one_model_call() -> None:
    images = torch.zeros((1, 3, 3, 1, 1))
    images[0, 0, 0, 0, 0] = 2.0
    present = torch.ones((1, 3), dtype=torch.bool)
    positions = torch.zeros((1, 3, 3))
    dataset = [(images, present, positions, torch.empty(0), torch.empty(0))]
    model = WindowLogitModel()

    predictions, labels, window_attention, slot_attention = predict(
        model, DataLoader(dataset, batch_size=1), torch.device("cpu")
    )

    expected = torch.sigmoid(torch.tensor(2.0)).item()
    assert model.starts == [2.0]
    assert labels is None
    assert predictions.shape == (1, 12)
    assert np.allclose(predictions, expected)
    assert window_attention.shape == (1, 12, 1, 3)
    assert slot_attention.shape == (1, 12, 1)


def test_dicom_cache_stores_normalized_images_as_uint8(tmp_path) -> None:
    cache_path = tmp_path / "study.npz"
    images = torch.tensor([0.0, 0.5, 1.0]).reshape(3, 1, 1, 1)
    present = torch.tensor([True, True, True])

    KneeStudyDataset._write_cache(cache_path, images, present)

    with np.load(cache_path, allow_pickle=False) as cached:
        assert cached["images"].dtype == np.uint8
        assert cached["images"].reshape(-1).tolist() == [0, 128, 255]
        assert cached["present"].tolist() == [True, True, True]
