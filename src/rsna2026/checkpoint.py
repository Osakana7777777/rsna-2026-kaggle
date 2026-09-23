"""Locate and verify a trained-model artifact mounted by Kaggle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any


CHECKPOINT_FORMAT_VERSION = 1


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_input_file(configured_path: str | Path, input_root: str | Path) -> Path:
    """Resolve a configured input across Kaggle-compatible mount roots."""
    configured = Path(configured_path)
    if configured.is_file():
        return configured
    root = Path(input_root)
    kaggle_input_root = Path("/kaggle/input")
    try:
        relative = configured.relative_to(kaggle_input_root)
    except ValueError:
        relative = configured if not configured.is_absolute() else None
    if relative is not None:
        remapped = root / relative
        if remapped.is_file():
            return remapped
    candidates = []
    for directory, names, files in os.walk(root):
        names[:] = [
            name for name in names if name not in {"train_series", "test_series"}
        ]
        if configured.name in files:
            candidates.append(Path(directory) / configured.name)
    candidates.sort()
    if len(candidates) != 1:
        raise FileNotFoundError(
            f"Expected one mounted {configured.name} beneath {root}, "
            f"found {candidates}"
        )
    return candidates[0]


def write_checkpoint_manifest(
    checkpoint_path: str | Path,
    manifest_path: str | Path,
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Write a sidecar manifest binding metadata to exact checkpoint bytes."""
    checkpoint = Path(checkpoint_path)
    manifest = {
        "format_version": CHECKPOINT_FORMAT_VERSION,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": file_sha256(checkpoint),
        **metadata,
    }
    Path(manifest_path).write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def load_and_verify_manifest(
    checkpoint_path: str | Path, manifest_path: str | Path
) -> dict[str, Any]:
    """Load a manifest and verify the mounted checkpoint's name and digest."""
    checkpoint = Path(checkpoint_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    if manifest.get("format_version") != CHECKPOINT_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported checkpoint format: {manifest.get('format_version')}"
        )
    if manifest.get("checkpoint_file") != checkpoint.name:
        raise ValueError("Checkpoint file name does not match its manifest")
    actual_sha256 = file_sha256(checkpoint)
    if manifest.get("checkpoint_sha256") != actual_sha256:
        raise ValueError("Checkpoint SHA-256 does not match its manifest")
    return manifest
