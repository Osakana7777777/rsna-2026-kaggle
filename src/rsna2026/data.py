"""Kaggle input-mount discovery and validation."""

from __future__ import annotations

from pathlib import Path


PLACEHOLDER = "YOUR_RSNA_2026_COMPETITION_SLUG"


def competition_input(input_root: str | Path, input_slug: str) -> Path:
    """Return the read-only competition mount after validating it exists."""
    if not input_slug or input_slug == PLACEHOLDER:
        raise ValueError(
            "Set competition.input_slug in configs/baseline.toml before running."
        )

    root = Path(input_root)
    candidates = (root / input_slug, root / "competitions" / input_slug)
    for path in candidates:
        if path.is_dir():
            return path

    available = sorted(item.name for item in root.iterdir()) if root.is_dir() else []
    raise FileNotFoundError(
        "Competition input mount not found. "
        f"Checked: {[str(path) for path in candidates]}. "
        f"Available mounts: {available}"
    )


def summarize_mount(path: Path, limit: int = 20) -> list[str]:
    """Return a small relative file listing without reading image data."""
    files = (item.relative_to(path).as_posix() for item in path.rglob("*") if item.is_file())
    return sorted(files)[:limit]
