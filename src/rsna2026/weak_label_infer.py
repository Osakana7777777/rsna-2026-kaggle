"""Deterministic vLLM batch inference for report-derived weak labels."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import subprocess
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rsna2026.weak_label_decode import decode_prediction, semantic_warnings
from rsna2026.weak_label_schema import VLLM_RSNA_LABEL_SCHEMA

LOGGER = logging.getLogger(__name__)
DEFAULT_MODEL = "Qwen/Qwen3.8-27B-FP8"
PROMPT_VERSION = "v1"
DEFAULT_PROMPT_PATH = (
    Path(__file__).resolve().parents[2] / "prompts" / "rsna_knee_extractor_v1.txt"
)


@dataclass(frozen=True)
class InferenceSettings:
    model: str = DEFAULT_MODEL
    model_revision: str | None = None
    model_path: str | None = None
    max_model_len: int = 65_536
    max_tokens: int = 1_200
    batch_size: int = 64
    max_num_seqs: int = 64
    gpu_memory_utilization: float = 0.92
    kv_cache_dtype: str = "fp8"
    enable_prefix_caching: bool = True
    generation_config: str = "auto"
    thinking: bool = False


@dataclass(frozen=True)
class Report:
    study_id: str
    text: str


def user_prompt(report: Report) -> str:
    return f"StudyInstanceUID:\n{report.study_id}\n\nRadiology report:\n{report.text}"


def messages(system_prompt: str, report: Report) -> list[dict[str, str]]:
    """Keep the shared prefix fixed and all case-specific text at the end."""
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt(report)},
    ]


def read_reports(path: Path, id_column: str, report_column: str) -> list[Report]:
    reports: list[Report] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Input CSV has no header: {path}")
        missing = {id_column, report_column}.difference(reader.fieldnames)
        if missing:
            raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
        for line_number, row in enumerate(reader, start=2):
            study_id = str(row[id_column] or "").strip()
            if not study_id:
                raise ValueError(f"Missing study identifier on CSV line {line_number}")
            if study_id in seen:
                raise ValueError(f"Duplicate study identifier: {study_id}")
            seen.add(study_id)
            reports.append(Report(study_id, str(row[report_column] or "")))
    return reports


def completed_study_ids(
    path: Path,
    expected_inference_settings: dict[str, Any] | None = None,
) -> set[str]:
    completed: set[str] = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                study_id = str(row["study_id"])
                decode_prediction(row["prediction"])
                if (
                    expected_inference_settings is not None
                    and row.get("inference_settings")
                    != expected_inference_settings
                ):
                    raise ValueError(
                        "inference settings differ from the current run; "
                        "use a new output path"
                    )
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Cannot resume from invalid JSONL line {line_number}: {error}"
                ) from error
            if study_id in completed:
                raise ValueError(f"Duplicate study identifier in output: {study_id}")
            completed.add(study_id)
    return completed


def batches(values: Sequence[Report], size: int) -> Iterator[Sequence[Report]]:
    if size < 1:
        raise ValueError("batch size must be positive")
    for start in range(0, len(values), size):
        yield values[start : start + size]


def append_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for row in rows:
            file.write(
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
        file.flush()
        os.fsync(file.fileno())


def create_engine(settings: InferenceSettings) -> tuple[Any, Any]:
    """Import vLLM lazily so validation and post-processing stay lightweight."""
    try:
        from vllm import LLM
        from vllm.sampling_params import StructuredOutputsParams
    except ImportError as error:
        raise RuntimeError(
            "vLLM is required for inference; run this command in the Modal GPU image"
        ) from error

    engine = LLM(
        model=settings.model_path or settings.model,
        tensor_parallel_size=1,
        max_model_len=settings.max_model_len,
        max_num_seqs=settings.max_num_seqs,
        enable_prefix_caching=settings.enable_prefix_caching,
        kv_cache_dtype=settings.kv_cache_dtype,
        gpu_memory_utilization=settings.gpu_memory_utilization,
        generation_config=settings.generation_config,
        trust_remote_code=False,
    )
    sampling = engine.get_default_sampling_params()
    sampling.max_tokens = settings.max_tokens
    sampling.structured_outputs = StructuredOutputsParams(
        json=VLLM_RSNA_LABEL_SCHEMA
    )
    return engine, sampling


def generation_settings(sampling: Any, source: str) -> dict[str, Any]:
    """Record the effective model-provided sampling values used by vLLM."""
    return {
        "source": source,
        "temperature": sampling.temperature,
        "top_p": sampling.top_p,
        "top_k": sampling.top_k,
        "max_tokens": sampling.max_tokens,
    }


def _gpu_memory_bytes() -> int | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        used_mebibytes = sum(
            int(line.strip()) for line in result.stdout.splitlines() if line.strip()
        )
        return used_mebibytes * 1024 * 1024
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        pass
    try:
        import torch

        return (
            int(torch.cuda.max_memory_allocated())
            if torch.cuda.is_available()
            else None
        )
    except ImportError:
        return None


def _nearest_rank(values: Sequence[int], percentile: float) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, int(percentile * len(ordered) + 0.999) - 1))
    return ordered[index]


def run_inference(
    reports: Sequence[Report],
    output_path: Path,
    error_path: Path,
    system_prompt: str,
    prompt_version: str,
    settings: InferenceSettings,
) -> dict[str, Any]:
    started = time.monotonic()
    input_tokens = 0
    output_tokens = 0
    generated_token_counts: list[int] = []
    failures = 0
    warning_count = 0
    processed = 0

    if not reports:
        return {
            "number_of_reports": 0,
            "processed": 0,
            "failures": 0,
            "semantic_warnings": 0,
            "prompt_tokens": 0,
            "generated_tokens": 0,
            "generated_tokens_min": None,
            "generated_tokens_p50": None,
            "generated_tokens_p95": None,
            "generated_tokens_max": None,
            "elapsed_seconds": 0.0,
            "reports_per_second": 0.0,
            "input_tokens_per_second": 0.0,
            "output_tokens_per_second": 0.0,
            "prefix_caching_enabled": settings.enable_prefix_caching,
            "prefix_cache_hit_metrics": None,
            "gpu_max_memory_allocated_bytes": None,
        }

    engine, sampling = create_engine(settings)
    effective_generation_settings = generation_settings(
        sampling, settings.generation_config
    )

    for report_batch in batches(reports, settings.batch_size):
        batch_messages = [messages(system_prompt, report) for report in report_batch]
        try:
            results = engine.chat(
                batch_messages,
                sampling_params=sampling,
                use_tqdm=True,
                chat_template_kwargs={"enable_thinking": settings.thinking},
            )
        except Exception as error:
            LOGGER.exception("vLLM failed for a batch of %d reports", len(report_batch))
            append_jsonl(
                error_path,
                [
                    {
                        "study_id": report.study_id,
                        "raw_output": None,
                        "error": f"inference_error: {type(error).__name__}: {error}",
                    }
                    for report in report_batch
                ],
            )
            failures += len(report_batch)
            continue

        valid_rows: list[dict[str, Any]] = []
        error_rows: list[dict[str, Any]] = []
        for report, result in zip(report_batch, results):
            raw_output = None
            try:
                raw_output = result.outputs[0].text
                generated_count = len(result.outputs[0].token_ids or ())
                input_tokens += len(result.prompt_token_ids or ())
                output_tokens += generated_count
                generated_token_counts.append(generated_count)
                prediction = json.loads(raw_output)
                decoded = decode_prediction(prediction)
                warnings = semantic_warnings(decoded)
            except (
                AttributeError,
                IndexError,
                TypeError,
                ValueError,
                json.JSONDecodeError,
            ) as error:
                error_rows.append(
                    {
                        "study_id": report.study_id,
                        "raw_output": raw_output,
                        "error": f"validation_error: {type(error).__name__}: {error}",
                    }
                )
                failures += 1
                continue
            warning_count += bool(warnings)
            valid_rows.append(
                {
                    "study_id": report.study_id,
                    "prompt_version": prompt_version,
                    "model": settings.model,
                    "inference_settings": asdict(settings),
                    "generation_settings": effective_generation_settings,
                    "prediction": prediction,
                    "raw_output": raw_output,
                    "semantic_warning": bool(warnings),
                    "semantic_warnings": warnings,
                }
            )
            processed += 1
        if len(results) != len(report_batch):
            missing_reports = report_batch[len(results) :]
            error_rows.extend(
                {
                    "study_id": report.study_id,
                    "raw_output": None,
                    "error": "inference_error: vLLM returned no result",
                }
                for report in missing_reports
            )
            failures += len(missing_reports)
        if valid_rows:
            append_jsonl(output_path, valid_rows)
        if error_rows:
            append_jsonl(error_path, error_rows)

    elapsed = time.monotonic() - started
    return {
        "number_of_reports": len(reports),
        "processed": processed,
        "failures": failures,
        "semantic_warnings": warning_count,
        "prompt_tokens": input_tokens,
        "generated_tokens": output_tokens,
        "generated_tokens_min": min(generated_token_counts, default=None),
        "generated_tokens_p50": _nearest_rank(generated_token_counts, 0.50),
        "generated_tokens_p95": _nearest_rank(generated_token_counts, 0.95),
        "generated_tokens_max": max(generated_token_counts, default=None),
        "elapsed_seconds": elapsed,
        "reports_per_second": processed / elapsed if elapsed else 0.0,
        "input_tokens_per_second": input_tokens / elapsed if elapsed else 0.0,
        "output_tokens_per_second": output_tokens / elapsed if elapsed else 0.0,
        "prefix_caching_enabled": settings.enable_prefix_caching,
        "prefix_cache_hit_metrics": None,
        "gpu_max_memory_allocated_bytes": _gpu_memory_bytes(),
        "generation_settings": effective_generation_settings,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument("--id-col", default="StudyInstanceUID")
    parser.add_argument("--report-col", default="report")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--prompt-version", default=PROMPT_VERSION)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision")
    parser.add_argument(
        "--model-path",
        help="Optional local snapshot path; model remains the metadata identity",
    )
    parser.add_argument("--max-model-len", type=int, default=65_536)
    parser.add_argument("--max-tokens", type=int, default=1_200)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-num-seqs", type=int, default=64)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument("--kv-cache-dtype", default="fp8")
    parser.add_argument(
        "--enable-prefix-caching",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    settings = InferenceSettings(
        model=args.model,
        model_revision=args.model_revision,
        model_path=args.model_path,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        batch_size=args.batch_size,
        max_num_seqs=args.max_num_seqs,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        enable_prefix_caching=args.enable_prefix_caching,
    )
    if not 0.0 < settings.gpu_memory_utilization <= 1.0:
        raise ValueError("gpu memory utilization must be in (0, 1]")
    if settings.max_num_seqs < 1:
        raise ValueError("max num seqs must be positive")
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    reports = read_reports(args.input, args.id_col, args.report_col)
    completed = (
        completed_study_ids(args.output, asdict(settings)) if args.resume else set()
    )
    if args.output.exists() and not args.resume and not args.dry_run:
        raise FileExistsError(f"Output already exists; use --resume: {args.output}")
    pending = [report for report in reports if report.study_id not in completed]
    LOGGER.info(
        "Loaded %d reports; %d complete and %d pending",
        len(reports),
        len(completed),
        len(pending),
    )
    if args.dry_run:
        LOGGER.info(
            "Dry run passed; prompt characters=%d, first user prompt=%r",
            len(system_prompt),
            user_prompt(pending[0]) if pending else None,
        )
        return 0

    error_path = args.error_output or args.output.with_suffix(".errors.jsonl")
    metrics_path = args.metrics_output or args.output.with_suffix(".metrics.json")
    metrics = run_inference(
        pending,
        args.output,
        error_path,
        system_prompt,
        args.prompt_version,
        settings,
    )
    metrics.update(
        {
            "prompt_version": args.prompt_version,
            "model": settings.model,
            "inference_settings": asdict(settings),
            "resumed_studies": len(completed),
        }
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    LOGGER.info("Inference metrics: %s", json.dumps(metrics, sort_keys=True))
    return 1 if metrics["failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
