"""Kaggle launcher with project files embedded by scripts/prepare.sh."""

from __future__ import annotations

import contextlib
import hashlib
import os
import shutil
import subprocess
import sys
import traceback
from pathlib import Path
from typing import TextIO


EMBEDDED_FILES: dict[str, str] = {}
RUN_MODE = "__RUN_MODE__"
CONFIG_RELATIVE_PATH = "configs/baseline.toml"


def prepare_payload() -> Path:
    """Materialize files embedded in the single script accepted by Kaggle."""
    if not EMBEDDED_FILES:
        return Path(__file__).resolve().parent

    payload_root = Path("/tmp/rsna2026-payload")
    for relative_path, contents in EMBEDDED_FILES.items():
        destination = payload_root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(contents, encoding="utf-8")
    return payload_root


PAYLOAD_ROOT = prepare_payload()
sys.path.insert(0, str(PAYLOAD_ROOT / "src"))

from rsna2026.config import load_config  # noqa: E402
from rsna2026.train import main as train_main  # noqa: E402


class Tee:
    """Write console output to both Kaggle's console and a retrievable log."""

    def __init__(self, *streams: TextIO) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()

    def isatty(self) -> bool:
        """Report a non-interactive stream for libraries that format console output."""
        return False


def run_payload() -> int:
    """Run EDA/inference directly or launch two distributed training processes."""
    if RUN_MODE == "eda":
        from rsna2026.eda import main as eda_main

        return eda_main([
            "--output-dir",
            "/kaggle/working/outputs/eda",
        ])

    image_cache_modes = {
        f"preprocess-{index}": index for index in range(4)
    }
    selected_mode = (
        "prepare-image-cache" if RUN_MODE in image_cache_modes else RUN_MODE
    )
    arguments = [
        "--config",
        str(PAYLOAD_ROOT / CONFIG_RELATIVE_PATH),
        "--mode",
        selected_mode,
    ]
    if RUN_MODE in image_cache_modes:
        arguments.extend(
            [
                "--cache-shard-index",
                str(image_cache_modes[RUN_MODE]),
                "--cache-shard-count",
                str(len(image_cache_modes)),
            ]
        )
    if RUN_MODE != "train":
        return train_main(arguments)

    environment = os.environ.copy()
    config = load_config(arguments[1])
    data_config = config["data"]
    image_cache_config = config.get("image_cache", {})
    persistent_image_cache = (
        isinstance(image_cache_config, dict)
        and bool(image_cache_config.get("enabled", False))
    )
    configured_cache = str(data_config.get("study_records_cache_path", ""))
    configured_sha256 = str(data_config.get("study_records_cache_sha256", ""))
    if bool(configured_cache) != bool(configured_sha256):
        raise ValueError(
            "Configured study-record cache path and SHA-256 must be set together"
        )
    if not configured_cache and not persistent_image_cache:
        preparation_arguments = [
            "--config",
            arguments[1],
            "--mode",
            "prepare-selection-cache",
        ]
        preparation_exit_code = train_main(preparation_arguments)
        if preparation_exit_code:
            return preparation_exit_code
        generated_cache = Path("/kaggle/working/outputs/slice-selection.json")
        runtime_cache = Path("/tmp/rsna2026-slice-selection.json")
        shutil.copyfile(generated_cache, runtime_cache)
        cache_sha256 = hashlib.sha256(runtime_cache.read_bytes()).hexdigest()
        environment["RSNA2026_STUDY_RECORDS_CACHE_PATH"] = str(runtime_cache)
        environment["RSNA2026_STUDY_RECORDS_CACHE_SHA256"] = cache_sha256
        print(
            "Starting two-GPU training with the newly prepared physical-window "
            f"selection cache: sha256={cache_sha256}",
            flush=True,
        )
    elif persistent_image_cache:
        print(
            "Starting two-GPU training from the pinned persistent image cache",
            flush=True,
        )
    source_root = str(PAYLOAD_ROOT / "src")
    current_python_path = environment.get("PYTHONPATH")
    environment["PYTHONPATH"] = (
        source_root
        if not current_python_path
        else os.pathsep.join((source_root, current_python_path))
    )
    environment["PYTHONUNBUFFERED"] = "1"
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        "-m",
        "rsna2026.train",
        *arguments,
    ]
    print(f"Launching two-GPU training: {' '.join(command)}", flush=True)
    process = subprocess.Popen(
        command,
        cwd=PAYLOAD_ROOT,
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    if process.stdout is None:
        raise RuntimeError("Distributed training output stream is unavailable")
    for line in process.stdout:
        print(line, end="", flush=True)
    return process.wait()


if __name__ == "__main__":
    output_dir = Path("/kaggle/working/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            with contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
                try:
                    exit_code = run_payload()
                    submission = output_dir / "submission.csv"
                    competition_submission = Path("/kaggle/working/submission.csv")
                    if exit_code == 0 and submission.is_file():
                        shutil.copyfile(submission, competition_submission)
                        print(f"Wrote {competition_submission}")
                except BaseException:  # Preserve a remote traceback in run.log.
                    traceback.print_exc()
                    exit_code = 1
    if exit_code:
        raise SystemExit(exit_code)
