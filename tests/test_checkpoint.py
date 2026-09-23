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


def test_resolve_input_file_remaps_kaggle_layout_before_name_fallback(tmp_path):
    expected = (
        tmp_path
        / "rsna-2026-preprocess-shard-2"
        / "outputs"
        / "image-cache"
        / "cache-manifest.json"
    )
    expected.parent.mkdir(parents=True)
    expected.write_bytes(b"expected")
    duplicate = (
        tmp_path
        / "rsna-2026-preprocess-shard-3"
        / "outputs"
        / "image-cache"
        / "cache-manifest.json"
    )
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes(b"duplicate")

    resolved = resolve_input_file(
        "/kaggle/input/rsna-2026-preprocess-shard-2/outputs/"
        "image-cache/cache-manifest.json",
        tmp_path,
    )

    assert resolved == expected


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
