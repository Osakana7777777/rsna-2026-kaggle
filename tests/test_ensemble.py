from __future__ import annotations

import json

import pytest

np = pytest.importorskip("numpy")

from rsna2026.checkpoint import file_sha256, write_checkpoint_manifest
from rsna2026.ensemble import load_ensemble_manifest, percentile_rank_mean


def test_percentile_rank_mean_handles_ties_per_target() -> None:
    first = np.asarray([[0.1, 0.5], [0.2, 0.5], [0.3, 0.9]])
    second = np.asarray([[0.9, 0.2], [0.8, 0.4], [0.7, 0.6]])

    result = percentile_rank_mean([first, second])

    assert np.allclose(result[:, 0], [2 / 3, 2 / 3, 2 / 3])
    assert np.allclose(result[:, 1], [5 / 12, 7 / 12, 1.0])


def test_manifest_verifies_four_checkpoint_hashes(tmp_path) -> None:
    members = []
    for fold in range(4):
        checkpoint = tmp_path / f"fold-{fold}" / "best-model.pt"
        checkpoint.parent.mkdir()
        checkpoint.write_bytes(f"fold {fold}".encode())
        checkpoint_manifest = checkpoint.parent / "checkpoint-manifest.json"
        write_checkpoint_manifest(
            checkpoint,
            checkpoint_manifest,
            {"fold": fold, "model": {"name": "example"}},
        )
        members.append(
            {
                "fold": fold,
                "checkpoint_file": f"fold-{fold}/best-model.pt",
                "checkpoint_sha256": file_sha256(checkpoint),
                "checkpoint_manifest_file": (
                    f"fold-{fold}/checkpoint-manifest.json"
                ),
            }
        )
    manifest_path = tmp_path / "ensemble-manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "format_version": 1,
                "model": {"name": "example"},
                "members": members,
            }
        ),
        encoding="utf-8",
    )

    loaded = load_ensemble_manifest(manifest_path)
    assert len(loaded["members"]) == 4

    (tmp_path / "fold-2" / "best-model.pt").write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        load_ensemble_manifest(manifest_path)
