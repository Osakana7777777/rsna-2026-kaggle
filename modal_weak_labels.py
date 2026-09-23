"""Run weak-label extraction on one Modal A100 80GB.

The input CSV must already exist in the ``rsna-weak-label-data`` Modal Volume.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal

app = modal.App("rsna-knee-weak-labels")

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "vllm==0.29.0",
        "transformers==5.17.0",
        "pyarrow>=17,<22",
    )
    .env(
        {
            "PYTHONPATH": "/root/src",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_dir("src", remote_path="/root/src")
    .add_local_dir("prompts", remote_path="/root/prompts")
)

prepare_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "huggingface_hub[hf_xet]>=0.34,<2",
)

data_volume = modal.Volume.from_name(
    "rsna-weak-label-data",
    create_if_missing=True,
)
model_cache = modal.Volume.from_name(
    "rsna-weak-label-model-cache",
    create_if_missing=True,
)


def _volume_path(relative_path: str) -> str:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError(f"Path must be relative to the data volume: {relative_path!r}")
    return str(PurePosixPath("/data") / path)


def _model_manifest_path(model: str, revision: str | None) -> Path:
    identity = f"{model}@{revision or 'default'}"
    digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()
    return Path("/root/.cache/huggingface/rsna-prepared-models") / f"{digest}.json"


@app.function(
    image=prepare_image,
    cpu=4,
    memory=8192,
    timeout=6 * 60 * 60,
    volumes={"/root/.cache/huggingface": model_cache},
)
def prepare_model(model: str, revision: str | None = None) -> str:
    """Download a complete model snapshot without allocating a GPU."""
    from huggingface_hub import snapshot_download

    snapshot_path = snapshot_download(
        repo_id=model,
        revision=revision,
        cache_dir="/root/.cache/huggingface",
    )
    manifest_path = _model_manifest_path(model, revision)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "model": model,
                "revision": revision,
                "snapshot_path": snapshot_path,
                "prepared_at": datetime.now(timezone.utc).isoformat(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    model_cache.commit()
    return snapshot_path


def _prepared_snapshot(model: str, revision: str | None) -> str:
    manifest_path = _model_manifest_path(model, revision)
    if not manifest_path.is_file():
        raise RuntimeError(
            "Model is not prepared in the shared cache. Run the prepare-model "
            "action before starting the A100 80GB inference function."
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != model or manifest.get("revision") != revision:
        raise RuntimeError(f"Prepared model manifest does not match {model!r}")
    snapshot_path = str(manifest.get("snapshot_path", ""))
    if not snapshot_path or not Path(snapshot_path).is_dir():
        raise RuntimeError(f"Prepared model snapshot is missing: {snapshot_path!r}")
    return snapshot_path


@app.function(
    image=prepare_image,
    cpu=1,
    memory=512,
    timeout=5 * 60,
    volumes={
        "/data": data_volume,
        "/root/.cache/huggingface": model_cache,
    },
)
def check_prepared_model(
    model: str,
    revision: str | None = None,
    input_path: str | None = None,
) -> str:
    """Validate cached model and input on CPU before requesting an A100 80GB."""
    snapshot_path = _prepared_snapshot(model, revision)
    if input_path is not None:
        mounted_input = Path(_volume_path(input_path))
        if not mounted_input.is_file():
            raise FileNotFoundError(
                f"Input does not exist in rsna-weak-label-data: {input_path!r}"
            )
    return snapshot_path


@app.function(
    image=image,
    gpu="A100-80GB",
    timeout=24 * 60 * 60,
    volumes={
        "/data": data_volume,
        "/root/.cache/huggingface": model_cache,
    },
)
def extract(
    input_path: str,
    output_path: str,
    id_column: str,
    report_column: str,
    model: str,
    revision: str | None,
    max_model_len: int,
    max_tokens: int,
    batch_size: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    dtype: str,
    kv_cache_dtype: str,
    thinking: bool,
    resume: bool,
) -> None:
    snapshot_path = _prepared_snapshot(model, revision)
    output = _volume_path(output_path)
    command = [
        "python",
        "-m",
        "rsna2026.weak_label_infer",
        "--input",
        _volume_path(input_path),
        "--output",
        output,
        "--id-col",
        id_column,
        "--report-col",
        report_column,
        "--prompt",
        "/root/prompts/rsna_knee_extractor_v1.txt",
        "--model",
        model,
        "--model-path",
        snapshot_path,
        "--max-model-len",
        str(max_model_len),
        "--max-tokens",
        str(max_tokens),
        "--batch-size",
        str(batch_size),
        "--max-num-seqs",
        str(max_num_seqs),
        "--gpu-memory-utilization",
        str(gpu_memory_utilization),
        "--dtype",
        dtype,
        "--kv-cache-dtype",
        kv_cache_dtype,
        "--enable-prefix-caching",
        "--thinking" if thinking else "--no-thinking",
    ]
    if revision is not None:
        command.extend(["--model-revision", revision])
    if resume:
        command.append("--resume")
    inference = subprocess.run(command, check=False)
    if Path(output).exists():
        subprocess.run(
            [
                "python",
                "-m",
                "rsna2026.weak_label_postprocess",
                "--input",
                output,
                "--output",
                _volume_path(str(PurePosixPath(output_path).with_suffix(".parquet"))),
            ],
            check=True,
        )
    data_volume.commit()
    inference.check_returncode()


@app.local_entrypoint()
def main(
    action: str = "infer",
    input_path: str = "reports.csv",
    output_path: str = "outputs/raw_predictions.jsonl",
    id_column: str = "StudyInstanceUID",
    report_column: str = "report",
    model: str = "google/gemma-4-12B-it",
    revision: str | None = None,
    max_model_len: int = 8_192,
    max_tokens: int = 4_096,
    batch_size: int = 32,
    max_num_seqs: int = 32,
    gpu_memory_utilization: float = 0.92,
    dtype: str = "bfloat16",
    kv_cache_dtype: str = "auto",
    thinking: bool = True,
    resume: bool = True,
) -> None:
    if action == "prepare-model":
        snapshot_path = prepare_model.remote(model=model, revision=revision)
        print(f"Prepared model snapshot on CPU: {snapshot_path}")
        return
    if action != "infer":
        raise ValueError("action must be 'prepare-model' or 'infer'")
    check_prepared_model.remote(
        model=model,
        revision=revision,
        input_path=input_path,
    )
    extract.remote(
        input_path=input_path,
        output_path=output_path,
        id_column=id_column,
        report_column=report_column,
        model=model,
        revision=revision,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        batch_size=batch_size,
        max_num_seqs=max_num_seqs,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        thinking=thinking,
        resume=resume,
    )
