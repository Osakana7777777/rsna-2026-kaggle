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


def test_launcher_uses_torchrun_only_for_distributed_training(monkeypatch):
    import io
    import os
    import types

    source = (PROJECT_ROOT / "kaggle/launcher.py").read_text()
    function = next(
        node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "run_payload"
    )
    code = compile(ast.Module(body=[function], type_ignores=[]), "launcher", "exec")
    calls = []

    class Process:
        stdout = io.StringIO("worker log\n")

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def wait(self):
            return 7

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    for distributed in (False, True):
        config = {
            "data": {}, "image_cache": {"enabled": True},
            "training": {"distributed": distributed, "expected_gpu_count": 2},
        }
        namespace = {
            "RUN_MODE": "train", "PAYLOAD_ROOT": PROJECT_ROOT,
            "CONFIG_RELATIVE_PATH": "configs/coatnet-0-224.toml",
            "load_config": lambda path: config, "train_main": lambda args: 0,
            "os": os, "sys": sys, "subprocess": types.SimpleNamespace(
                Popen=popen, PIPE=-1, STDOUT=-2,
            ),
        }
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        exec(code, namespace)
        result = namespace["run_payload"]()
        if distributed:
            assert result == 7
            command, options = calls[-1]
            assert "torch.distributed.run" in command
            assert "--nproc-per-node=2" in command
            assert "rsna2026.train" in command
            assert "CUDA_VISIBLE_DEVICES" not in options["env"]
            assert options["env"]["PYTHONPATH"].startswith(str(PROJECT_ROOT / "src"))
        else:
            assert result == 0
            assert os.environ["CUDA_VISIBLE_DEVICES"] == "0"
            assert not calls
