from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = PROJECT_ROOT / "scripts" / "build_launcher.py"


def read_launcher_values(source: str) -> dict[str, object]:
    wanted = {"EMBEDDED_FILES", "RUN_MODE", "CONFIG_RELATIVE_PATH"}
    values = {}
    for node in ast.parse(source).body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id in wanted
        ):
            values[node.targets[0].id] = ast.literal_eval(node.value)
        elif (
            isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id in wanted
        ):
            values[node.target.id] = ast.literal_eval(node.value)
    return values


def build_launcher(tmp_path: Path, config: str) -> dict[str, object]:
    output = tmp_path / "launcher.ipynb"
    subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--mode",
            "train",
            "--config",
            config,
            "--output",
            str(output),
        ],
        check=True,
    )
    notebook = json.loads(output.read_text(encoding="utf-8"))
    return read_launcher_values(str(notebook["cells"][0]["source"]))


def test_build_launcher_embeds_selected_config_only(tmp_path: Path) -> None:
    values = build_launcher(tmp_path, "configs/coatnet-0-224.toml")
    embedded = values["EMBEDDED_FILES"]

    assert values["CONFIG_RELATIVE_PATH"] == "configs/coatnet-0-224.toml"
    assert isinstance(embedded, dict)
    assert sorted(key for key in embedded if key.startswith("configs/")) == [
        "configs/coatnet-0-224.toml"
    ]


def test_build_launcher_defaults_to_baseline(tmp_path: Path) -> None:
    output = tmp_path / "launcher.ipynb"
    subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--mode",
            "train",
            "--output",
            str(output),
        ],
        check=True,
    )
    notebook = json.loads(output.read_text(encoding="utf-8"))
    values = read_launcher_values(str(notebook["cells"][0]["source"]))
    embedded = values["EMBEDDED_FILES"]

    assert values["CONFIG_RELATIVE_PATH"] == "configs/baseline.toml"
    assert isinstance(embedded, dict)
    assert sorted(key for key in embedded if key.startswith("configs/")) == [
        "configs/baseline.toml"
    ]


def test_build_launcher_rejects_config_outside_configs(tmp_path: Path) -> None:
    outside = tmp_path / "outside.toml"
    outside.write_text("[run]\nseed = 1\n", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(BUILD_SCRIPT),
            "--project-root",
            str(PROJECT_ROOT),
            "--mode",
            "train",
            "--config",
            str(outside),
            "--output",
            str(tmp_path / "launcher.ipynb"),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "Configuration must be below" in result.stderr
