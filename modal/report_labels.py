"""Private Modal application for Gemma 4 report-label generation."""

# ruff: noqa: I001

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import modal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna2026.constants import TARGETS
from rsna2026.public_labels import file_sha256
from rsna2026.report_labels import (
    MODEL_REPOSITORY,
    MODEL_REVISION,
    TRANSFORMERS_VERSION,
    VLLM_VERSION,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_jsonl,
    build_audit,
    build_messages,
    empty_report_labels,
    file_manifest,
    flatten_audit_rows,
    group_reports,
    load_label_config,
    load_public_probabilities,
    output_json_schema,
    probability_for,
    prompt_hash,
    read_train_csv,
    report_hash,
    sha256_jsonable,
    validate_checkpoint_metadata,
    validate_response,
)


APP_NAME = "rsna-2026-gemma4-report-labels"
HF_CACHE_PATH = "/root/.cache/huggingface"
VLLM_CACHE_PATH = "/root/.cache/vllm"
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "report-labels.toml"
CONFIG = load_label_config(DEFAULT_CONFIG)
MODEL_CONFIG = CONFIG["model"]

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.9.0-devel-ubuntu22.04", add_python="3.12"
    )
    .entrypoint([])
    .uv_pip_install(
        f"vllm=={VLLM_VERSION}", f"transformers=={TRANSFORMERS_VERSION}"
    )
    .env(
        {
            "HF_HOME": HF_CACHE_PATH,
            "HF_XET_HIGH_PERFORMANCE": "1",
            "VLLM_CACHE_ROOT": VLLM_CACHE_PATH,
            "PYTHONPATH": "/root/src",
        }
    )
    .add_local_dir(str(PROJECT_ROOT / "src"), remote_path="/root/src", copy=True)
    .add_local_file(
        str(DEFAULT_CONFIG), remote_path="/configs/report-labels.toml", copy=True
    )
)

app = modal.App(APP_NAME)
hf_cache = modal.Volume.from_name("rsna-2026-gemma4-hf-cache", create_if_missing=True)
vllm_cache = modal.Volume.from_name(
    "rsna-2026-gemma4-vllm-cache", create_if_missing=True
)


@app.cls(
    image=image,
    gpu="RTX-PRO-6000",
    timeout=60 * 60,
    startup_timeout=20 * 60,
    scaledown_window=15 * 60,
    max_containers=1,
    secrets=[modal.Secret.from_name(str(MODEL_CONFIG["huggingface_secret"]))],
    volumes={HF_CACHE_PATH: hf_cache, VLLM_CACHE_PATH: vllm_cache},
)
class GemmaReportLabeler:
    """One private, single-GPU vLLM offline inference worker."""

    @modal.enter()
    def start(self) -> None:
        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        self.runtime = {
            "gpu": torch.cuda.get_device_name(0),
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "vllm_version": vllm.__version__,
        }
        if vllm.__version__ != VLLM_VERSION:
            raise RuntimeError(
                f"Expected vLLM {VLLM_VERSION}, found {vllm.__version__}"
            )
        if transformers.__version__ != TRANSFORMERS_VERSION:
            raise RuntimeError(
                f"Expected Transformers {TRANSFORMERS_VERSION}, "
                f"found {transformers.__version__}"
            )
        self.llm = LLM(
            model=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            dtype=str(MODEL_CONFIG["dtype"]),
            max_model_len=int(MODEL_CONFIG["max_model_len"]),
            max_num_seqs=int(MODEL_CONFIG["max_num_seqs"]),
            gpu_memory_utilization=float(MODEL_CONFIG["gpu_memory_utilization"]),
            tensor_parallel_size=1,
            seed=int(CONFIG["run"]["seed"]),
            limit_mm_per_prompt={"image": 0, "video": 0, "audio": 0},
        )
        structured = StructuredOutputsParams(json=output_json_schema())
        self.sampling = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            n=1,
            seed=int(CONFIG["run"]["seed"]),
            max_tokens=int(MODEL_CONFIG["max_output_tokens"]),
            structured_outputs=structured,
        )

    @modal.method()
    def infer(self, requests: list[dict[str, object]]) -> dict[str, object]:
        conversations = [request["messages"] for request in requests]
        started = time.monotonic()
        outputs = self.llm.chat(
            conversations,
            sampling_params=self.sampling,
            chat_template_kwargs={"enable_thinking": False},
        )
        responses = []
        for request, output in zip(requests, outputs):
            responses.append(
                {
                    "report_hash": request["report_hash"],
                    "response": output.outputs[0].text,
                    "prompt_tokens": len(output.prompt_token_ids),
                    "output_tokens": len(output.outputs[0].token_ids),
                }
            )
        return {
            "responses": responses,
            "elapsed_seconds": time.monotonic() - started,
            "runtime": self.runtime,
        }


def _checkpoint_rows(path: Path, signature: str) -> dict[str, dict[str, object]]:
    metadata_path = path.parent / "checkpoint-metadata.json"
    if path.exists() != metadata_path.exists():
        raise ValueError("Checkpoint data and metadata must either both exist or both be absent")
    if not path.exists():
        atomic_write_json(metadata_path, {"signature": signature})
        return {}
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validate_checkpoint_metadata(metadata, signature)
    rows: dict[str, dict[str, object]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        digest = str(row["report_hash"])
        if digest in rows:
            raise ValueError(f"Duplicate checkpoint report hash: {digest}")
        rows[digest] = row
    return rows


def _write_checkpoint(path: Path, rows: dict[str, dict[str, object]]) -> None:
    atomic_write_jsonl(path, [rows[key] for key in sorted(rows)])


def _synthetic_rows() -> list[dict[str, str]]:
    rows = []
    reports = (
        "Complete ACL tear. No acute fracture. Moderate joint effusion.",
        "The ACL is intact. Trace joint fluid. Small popliteal cyst.",
    )
    for index, report in enumerate(reports):
        row = {"StudyInstanceUID": f"synthetic-{index}", "Report": report}
        row.update({target: "" for target in TARGETS})
        rows.append(row)
    return rows


@app.local_entrypoint()
def main(
    input_csv: str = "",
    public_labels: str = "",
    output_dir: str = "",
    config: str = str(DEFAULT_CONFIG),
    smoke: bool = False,
) -> None:
    """Generate, validate, resume, materialize, and optionally audit labels."""
    label_config = load_label_config(config)
    expected_studies = int(label_config["run"]["expected_studies"])
    if smoke:
        train_rows = _synthetic_rows()
        reports_only = [
            {"StudyInstanceUID": row["StudyInstanceUID"], "Report": row["Report"]}
            for row in train_rows
        ]
        input_hash = sha256_jsonable(reports_only)
        input_encoding = "synthetic"
    else:
        if not input_csv:
            raise ValueError("--input-csv is required unless --smoke is used")
        train_rows, reports_only, input_encoding = read_train_csv(
            input_csv, expected_studies
        )
        input_hash = file_sha256(Path(input_csv))
    public: dict[str, dict[str, float]] = {}
    if not smoke:
        if not public_labels:
            raise ValueError("--public-labels is required for the 58-study audit")
        public_config = label_config["public_labels"]
        public = load_public_probabilities(
            public_labels,
            str(public_config["sha256"]),
            int(public_config["expected_rows"]),
        )
    unique_reports, members = group_reports(reports_only)
    signature_payload = {
        "input_sha256": input_hash,
        "model": MODEL_REPOSITORY,
        "revision": MODEL_REVISION,
        "vllm": VLLM_VERSION,
        "prompt_sha256": prompt_hash(),
        "config_sha256": file_sha256(Path(config)),
    }
    signature = sha256_jsonable(signature_payload)
    run_id = f"gemma4-{signature[:12]}" + ("-smoke" if smoke else "")
    destination = Path(output_dir) if output_dir else Path(
        str(label_config["run"]["output_root"])
    ) / run_id
    destination.mkdir(parents=True, exist_ok=True)
    checkpoint_path = destination / "label-checkpoint.jsonl"
    checkpoint = _checkpoint_rows(checkpoint_path, signature)
    report_by_hash = {str(row["report_hash"]): str(row["report"]) for row in unique_reports}

    for digest, report in report_by_hash.items():
        if not report.strip() and digest not in checkpoint:
            checkpoint[digest] = {
                "report_hash": digest,
                "findings": empty_report_labels(),
                "retried": False,
                "prompt_tokens": 0,
                "output_tokens": 0,
            }
    _write_checkpoint(checkpoint_path, checkpoint)

    pending = [
        row for row in unique_reports if str(row["report_hash"]) not in checkpoint
    ]
    batch_size = int(label_config["run"]["batch_size"])
    failures: list[dict[str, object]] = []
    runtime: dict[str, object] = {}
    remote_elapsed = 0.0
    remote_reports = 0
    remote_prompt_tokens = 0
    remote_output_tokens = 0
    started = time.monotonic()
    worker = GemmaReportLabeler()
    for offset in range(0, len(pending), batch_size):
        batch = pending[offset : offset + batch_size]
        requests = [
            {
                "report_hash": row["report_hash"],
                "messages": build_messages(str(row["report"])),
            }
            for row in batch
        ]
        result = worker.infer.remote(requests)
        runtime = dict(result["runtime"])
        remote_elapsed += float(result["elapsed_seconds"])
        remote_reports += len(result["responses"])
        remote_prompt_tokens += sum(
            int(response["prompt_tokens"]) for response in result["responses"]
        )
        remote_output_tokens += sum(
            int(response["output_tokens"]) for response in result["responses"]
        )
        invalid: list[dict[str, object]] = []
        for response in result["responses"]:
            digest = str(response["report_hash"])
            try:
                findings = validate_response(
                    str(response["response"]), report_by_hash[digest]
                )
            except ValueError as error:
                invalid.append({**response, "error": str(error)})
                continue
            checkpoint[digest] = {
                "report_hash": digest,
                "findings": findings,
                "retried": False,
                "prompt_tokens": response["prompt_tokens"],
                "output_tokens": response["output_tokens"],
            }
        if invalid:
            retry_requests = [
                {
                    "report_hash": row["report_hash"],
                    "messages": build_messages(
                        report_by_hash[str(row["report_hash"])],
                        [str(row["error"])],
                        str(row["response"]),
                    ),
                }
                for row in invalid
            ]
            retried = worker.infer.remote(retry_requests)
            remote_elapsed += float(retried["elapsed_seconds"])
            remote_reports += len(retried["responses"])
            remote_prompt_tokens += sum(
                int(response["prompt_tokens"]) for response in retried["responses"]
            )
            remote_output_tokens += sum(
                int(response["output_tokens"]) for response in retried["responses"]
            )
            for response in retried["responses"]:
                digest = str(response["report_hash"])
                try:
                    findings = validate_response(
                        str(response["response"]), report_by_hash[digest]
                    )
                except ValueError as error:
                    failures.append(
                        {
                            "report_hash": digest,
                            "study_uids": members[digest],
                            "error": str(error),
                            "retried": True,
                        }
                    )
                    continue
                checkpoint[digest] = {
                    "report_hash": digest,
                    "findings": findings,
                    "retried": True,
                    "prompt_tokens": response["prompt_tokens"],
                    "output_tokens": response["output_tokens"],
                }
        _write_checkpoint(checkpoint_path, checkpoint)
        print(
            f"Completed {len(checkpoint)}/{len(unique_reports)} unique reports",
            flush=True,
        )

    atomic_write_jsonl(destination / "failed-labels.jsonl", failures)
    missing = sorted(set(report_by_hash).difference(checkpoint))
    if failures or missing:
        raise RuntimeError(
            f"Label generation is incomplete: failures={len(failures)}, missing={len(missing)}"
        )

    structured_rows: list[dict[str, object]] = []
    pseudo_rows: list[dict[str, object]] = []
    for row in reports_only:
        uid = str(row["StudyInstanceUID"])
        digest = report_hash(str(row["Report"]))
        record = checkpoint[digest]
        findings = record["findings"]
        probabilities = {
            target: probability_for(
                str(findings[target]["state"]),  # type: ignore[index]
                str(findings[target]["certainty"]),  # type: ignore[index]
                label_config,
            )
            for target in TARGETS
        }
        structured_rows.append(
            {
                "StudyInstanceUID": uid,
                "report_hash": digest,
                "retried": record["retried"],
                "findings": findings,
                "probabilities": probabilities,
            }
        )
        pseudo_rows.append({"StudyInstanceUID": uid, **probabilities})
    if len(structured_rows) != len(reports_only):
        raise RuntimeError("Not every study was expanded from its report group")
    if any(
        not math.isfinite(float(row[target])) or not 0.0 <= float(row[target]) <= 1.0
        for row in pseudo_rows
        for target in TARGETS
    ):
        raise RuntimeError("A generated probability is outside [0, 1]")

    structured_path = destination / "structured-labels.jsonl"
    pseudo_path = destination / "pseudo-labels.csv"
    atomic_write_jsonl(structured_path, structured_rows)
    atomic_write_csv(pseudo_path, pseudo_rows, ["StudyInstanceUID", *TARGETS])

    audit_paths: list[Path] = []
    audit_summary: dict[str, object] = {}
    if not smoke:
        audit_summary, audit_rows = build_audit(
            train_rows, structured_rows, public, label_config
        )
        audit_json_path = destination / "label-audit.json"
        audit_csv_path = destination / "label-audit.csv"
        atomic_write_json(audit_json_path, audit_summary)
        flat_audit = flatten_audit_rows(audit_rows)
        atomic_write_csv(audit_csv_path, flat_audit, list(flat_audit[0]))
        audit_paths.extend((audit_json_path, audit_csv_path))

    elapsed = time.monotonic() - started
    hashed_paths = [
        checkpoint_path,
        destination / "checkpoint-metadata.json",
        structured_path,
        pseudo_path,
        destination / "failed-labels.jsonl",
        *audit_paths,
    ]
    summary = {
        "status": "complete",
        "run_id": run_id,
        **signature_payload,
        "input_encoding": input_encoding,
        "studies": len(reports_only),
        "unique_reports": len(unique_reports),
        "duplicate_studies_saved": len(reports_only) - len(unique_reports),
        "retried_unique_reports": sum(bool(row["retried"]) for row in checkpoint.values()),
        "semantic_validation_rate": 1.0,
        "elapsed_seconds": elapsed,
        "remote_inference_seconds": remote_elapsed,
        "remote_inference_speed": {
            "requests": remote_reports,
            "prompt_tokens": remote_prompt_tokens,
            "output_tokens": remote_output_tokens,
            "requests_per_second": (
                remote_reports / remote_elapsed if remote_elapsed else None
            ),
            "tokens_per_second": (
                (remote_prompt_tokens + remote_output_tokens) / remote_elapsed
                if remote_elapsed
                else None
            ),
        },
        "runtime": runtime,
        "audit": audit_summary,
        "artifact_sha256": file_manifest(hashed_paths),
    }
    summary_path = destination / "run-summary.json"
    atomic_write_json(summary_path, summary)
    atomic_write_json(
        destination / "manifest.json",
        file_manifest([*hashed_paths, summary_path]),
    )
    print(f"Completed report labeling in {destination}", flush=True)
