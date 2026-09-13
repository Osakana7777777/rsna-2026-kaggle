"""Build the single Python source file uploaded by the Kaggle CLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence


EMBEDDED_FILES_MARKER = "EMBEDDED_FILES: dict[str, str] = {}"
RUN_MODE_MARKER = 'RUN_MODE = "__RUN_MODE__"'
CONFIG_PATH_MARKER = 'CONFIG_RELATIVE_PATH = "configs/baseline.toml"'
RUN_MODES = (
    "train",
    "inference",
    "eda",
    "preprocess-0",
    "preprocess-1",
    "preprocess-2",
    "preprocess-3",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=RUN_MODES, required=True)
    parser.add_argument(
        "--config", type=Path, default=Path("configs/baseline.toml")
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    project_root = args.project_root.resolve()
    config_root = (project_root / "configs").resolve()
    config_path = (
        args.config.resolve()
        if args.config.is_absolute()
        else (project_root / args.config).resolve()
    )
    try:
        config_path.relative_to(config_root)
    except ValueError as error:
        raise ValueError(
            f"Configuration must be below {config_root}: {config_path}"
        ) from error
    if config_path.suffix != ".toml" or not config_path.is_file():
        raise FileNotFoundError(f"Configuration TOML not found: {config_path}")
    config_relative_path = config_path.relative_to(project_root).as_posix()
    launcher = (project_root / "kaggle" / "launcher.py").read_text(encoding="utf-8")

    source_files = sorted((project_root / "src" / "rsna2026").rglob("*.py"))
    files = source_files + [config_path]
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
    if launcher.count(RUN_MODE_MARKER) != 1:
        raise ValueError("Launcher must contain exactly one run-mode marker")
    if launcher.count(CONFIG_PATH_MARKER) != 1:
        raise ValueError("Launcher must contain exactly one config-path marker")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    generated_source = launcher.replace(EMBEDDED_FILES_MARKER, replacement)
    generated_source = generated_source.replace(
        RUN_MODE_MARKER, f'RUN_MODE = "{args.mode}"'
    )
    generated_source = generated_source.replace(
        CONFIG_PATH_MARKER,
        f'CONFIG_RELATIVE_PATH = "{config_relative_path}"',
    )
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
