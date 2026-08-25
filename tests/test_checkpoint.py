import json

import pytest

from rsna2026.checkpoint import (
    load_and_verify_manifest,
    resolve_input_file,
    write_checkpoint_manifest,
)


def test_manifest_binds_checkpoint_bytes(tmp_path):
    checkpoint = tmp_path / "best-model.pt"
    manifest_path = tmp_path / "checkpoint-manifest.json"
    checkpoint.write_bytes(b"trained weights")

    written = write_checkpoint_manifest(
        checkpoint, manifest_path, {"model": {"name": "example"}}
    )

    assert load_and_verify_manifest(checkpoint, manifest_path) == written
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="SHA-256"):
        load_and_verify_manifest(checkpoint, manifest_path)


def test_resolve_input_file_uses_unique_name_fallback(tmp_path):
    mounted = tmp_path / "notebook-output" / "outputs" / "best-model.pt"
    mounted.parent.mkdir(parents=True)
    mounted.write_bytes(b"weights")

    resolved = resolve_input_file(
        "/kaggle/input/expected-slug/outputs/best-model.pt", tmp_path
    )

    assert resolved == mounted


def test_manifest_rejects_wrong_checkpoint_name(tmp_path):
    checkpoint = tmp_path / "best-model.pt"
    manifest_path = tmp_path / "checkpoint-manifest.json"
    checkpoint.write_bytes(b"weights")
    write_checkpoint_manifest(checkpoint, manifest_path, {})
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["checkpoint_file"] = "another.pt"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="file name"):
        load_and_verify_manifest(checkpoint, manifest_path)
