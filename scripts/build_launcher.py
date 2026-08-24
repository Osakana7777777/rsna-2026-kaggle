"""Build the single Python source file uploaded by the Kaggle CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


EMBEDDED_FILES_MARKER = "EMBEDDED_FILES: dict[str, str] = {}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    launcher = (project_root / "kaggle" / "launcher.py").read_text(encoding="utf-8")

    source_files = sorted((project_root / "src" / "rsna2026").rglob("*.py"))
    files = source_files + [project_root / "configs" / "baseline.toml"]
    embedded_files = {
        path.relative_to(project_root).as_posix(): path.read_text(encoding="utf-8")
        for path in files
    }
    replacement = (
        "EMBEDDED_FILES: dict[str, str] = "
        + json.dumps(embedded_files, ensure_ascii=True, sort_keys=True)
    )
    if launcher.count(EMBEDDED_FILES_MARKER) != 1:
        raise ValueError("Launcher must contain exactly one embedded-files marker")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_source = launcher.replace(EMBEDDED_FILES_MARKER, replacement)
    if args.output.suffix == ".ipynb":
        notebook = {
            "cells": [
                {
                    "cell_type": "code",
                    "id": "rsna2026-launcher",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": generated_source,
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3",
                },
                "language_info": {"name": "python", "version": "3.12"},
            },
            "nbformat": 4,
            "nbformat_minor": 5,
        }
        output = json.dumps(notebook, ensure_ascii=True)
    else:
        output = generated_source
    args.output.write_text(output, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
