"""Generate report-derived weak labels with the OpenAI Responses API."""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rsna2026.weak_label_decode import decode_prediction, semantic_warnings
from rsna2026.weak_label_infer import (
    DEFAULT_PROMPT_PATH,
    Report,
    append_jsonl,
    completed_study_ids,
    read_reports,
    user_prompt,
)
from rsna2026.weak_label_openai_schema_v2 import (
    OPENAI_RSNA_LABEL_SCHEMA_V2,
    OPENAI_SCHEMA_VERSION,
)
from rsna2026.weak_label_schema import (
    compact_openai_prediction,
    normalize_unique_items,
    validate_prediction,
)

LOGGER = logging.getLogger(__name__)
DEFAULT_OPENAI_MODEL = "gpt-5.6-luna"
DEFAULT_OPENAI_PROMPT_VERSION = "openai-v1"
REASONING_EFFORTS = ("none", "low", "medium", "high", "xhigh", "max")


@dataclass(frozen=True)
class OpenAIInferenceSettings:
    model: str = DEFAULT_OPENAI_MODEL
    reasoning_effort: str = "medium"
    max_output_tokens: int = 4_096
    workers: int = 8
    timeout_seconds: float = 180.0
    max_retries: int = 4
    store: bool = False


def openai_prompt(prompt: str) -> str:
    """Adapt only the wire representation while preserving extraction semantics."""
    return prompt.replace(
        "Each finding is exactly [m,c,v,a,t,k,r,e].",
        "Each finding is exactly an object with the named fields m,c,v,a,t,k,r,e.",
    )


def response_request(
    prompt: str,
    report: Report,
    settings: OpenAIInferenceSettings,
    prompt_version: str,
) -> dict[str, Any]:
    """Build one strict, cache-friendly Responses API request."""
    return {
        "model": settings.model,
        "input": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_text",
                        "text": openai_prompt(prompt),
                        "prompt_cache_breakpoint": {"mode": "explicit"},
                    }
                ],
            },
            {"role": "user", "content": user_prompt(report)},
        ],
        "reasoning": {"effort": settings.reasoning_effort},
        "max_output_tokens": settings.max_output_tokens,
        "text": {
            "format": {
                "type": "json_schema",
                "name": "rsna_knee_weak_label_v2",
                "strict": True,
                "schema": OPENAI_RSNA_LABEL_SCHEMA_V2,
            }
        },
        "prompt_cache_key": (
            f"rsna-knee-weak-label:{prompt_version}:{OPENAI_SCHEMA_VERSION}"
        ),
        "prompt_cache_options": {"mode": "explicit", "ttl": "30m"},
        "store": settings.store,
    }


def _response_output_text(response: Any) -> str:
    status = getattr(response, "status", None)
    if status != "completed":
        details = getattr(response, "incomplete_details", None)
        reason = getattr(details, "reason", None)
        raise ValueError(f"OpenAI response status is {status!r}: {reason!r}")
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text.strip():
        raise ValueError("OpenAI response contains no output text")
    return output_text.strip()


def _usage(response: Any) -> dict[str, int | None]:
    usage = getattr(response, "usage", None)
    details = getattr(usage, "input_tokens_details", None)
    return {
        "input_tokens": getattr(usage, "input_tokens", None),
        "cached_input_tokens": getattr(details, "cached_tokens", None),
        "cache_write_tokens": getattr(details, "cache_write_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    }


def generate_one(
    client: Any,
    report: Report,
    prompt: str,
    prompt_version: str,
    settings: OpenAIInferenceSettings,
) -> dict[str, Any]:
    """Generate, normalize, and fully validate one report extraction."""
    response = client.responses.create(
        **response_request(prompt, report, settings, prompt_version)
    )
    raw_output = _response_output_text(response)
    api_prediction = json.loads(raw_output)
    compact_prediction = compact_openai_prediction(api_prediction)
    prediction, normalized_paths = normalize_unique_items(compact_prediction)
    prediction = validate_prediction(prediction)
    warnings = semantic_warnings(decode_prediction(prediction))
    return {
        "study_id": report.study_id,
        "prompt_version": prompt_version,
        "schema_version": OPENAI_SCHEMA_VERSION,
        "model": settings.model,
        "inference_settings": asdict(settings),
        "generation_settings": {
            "api": "responses",
            "reasoning_effort": settings.reasoning_effort,
            "max_output_tokens": settings.max_output_tokens,
            "strict_json_schema": True,
            "schema_version": OPENAI_SCHEMA_VERSION,
        },
        "prediction": prediction,
        "raw_output": raw_output,
        "reasoning": None,
        "normalized_unique_items": normalized_paths,
        "semantic_warning": bool(warnings),
        "semantic_warnings": warnings,
        "openai_response_id": getattr(response, "id", None),
        "usage": _usage(response),
    }


def run_inference(
    reports: Sequence[Report],
    output_path: Path,
    error_path: Path,
    prompt: str,
    prompt_version: str,
    settings: OpenAIInferenceSettings,
) -> dict[str, Any]:
    """Run bounded concurrent API requests and persist each completed result."""
    if not reports:
        return {
            "number_of_reports": 0,
            "processed": 0,
            "failures": 0,
            "input_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "cached_input_token_rate": 0.0,
            "output_tokens": 0,
            "elapsed_seconds": 0.0,
            "reports_per_second": 0.0,
            "cache_warmup_reports": 0,
        }
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError(
            "Install the OpenAI Python SDK to run API inference"
        ) from error

    client = OpenAI(timeout=settings.timeout_seconds, max_retries=settings.max_retries)
    started = time.monotonic()
    processed = 0
    failures = 0
    input_tokens = 0
    cached_input_tokens = 0
    cache_write_tokens = 0
    output_tokens = 0

    def record_result(report: Report, result: Any) -> None:
        nonlocal processed, failures
        nonlocal input_tokens, cached_input_tokens, cache_write_tokens, output_tokens
        try:
            row = result()
        except Exception as error:
            LOGGER.exception("OpenAI generation failed for %s", report.study_id)
            append_jsonl(
                error_path,
                [
                    {
                        "study_id": report.study_id,
                        "raw_output": None,
                        "error": f"{type(error).__name__}: {error}",
                    }
                ],
            )
            failures += 1
        else:
            append_jsonl(output_path, [row])
            usage = row["usage"]
            input_tokens += usage["input_tokens"] or 0
            cached_input_tokens += usage["cached_input_tokens"] or 0
            cache_write_tokens += usage["cache_write_tokens"] or 0
            output_tokens += usage["output_tokens"] or 0
            processed += 1

    pending_reports = list(reports)
    if pending_reports:
        warmup_report = pending_reports.pop(0)
        record_result(
            warmup_report,
            lambda: generate_one(
                client,
                warmup_report,
                prompt,
                prompt_version,
                settings,
            ),
        )
    with ThreadPoolExecutor(max_workers=settings.workers) as executor:
        futures = {
            executor.submit(
                generate_one,
                client,
                report,
                prompt,
                prompt_version,
                settings,
            ): report
            for report in pending_reports
        }
        for future in as_completed(futures):
            record_result(futures[future], future.result)

    elapsed = time.monotonic() - started
    return {
        "number_of_reports": len(reports),
        "processed": processed,
        "failures": failures,
        "input_tokens": input_tokens,
        "cached_input_tokens": cached_input_tokens,
        "cache_write_tokens": cache_write_tokens,
        "cached_input_token_rate": (
            cached_input_tokens / input_tokens if input_tokens else 0.0
        ),
        "output_tokens": output_tokens,
        "elapsed_seconds": elapsed,
        "reports_per_second": processed / elapsed if elapsed else 0.0,
        "cache_warmup_reports": min(len(reports), 1),
    }


def read_error_study_ids(path: Path) -> list[str]:
    """Read unique study identifiers from an immutable error JSONL log."""
    if not path.is_file():
        raise FileNotFoundError(f"Error JSONL does not exist: {path}")
    study_ids: list[str] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                study_id = row["study_id"]
                if not isinstance(study_id, str) or not study_id.strip():
                    raise ValueError("study_id must be a non-empty string")
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid error JSONL line {line_number}: {error}"
                ) from error
            study_id = study_id.strip()
            if study_id not in seen:
                study_ids.append(study_id)
                seen.add(study_id)
    return study_ids


def select_retry_reports(
    reports: Sequence[Report],
    error_path: Path,
    output_path: Path,
    expected_inference_settings: dict[str, Any],
) -> tuple[list[Report], int, int]:
    """Select failed studies, excluding any already present in valid output."""
    failed_ids = read_error_study_ids(error_path)
    failed_set = set(failed_ids)
    report_ids = {report.study_id for report in reports}
    missing_ids = failed_set.difference(report_ids)
    if missing_ids:
        preview = sorted(missing_ids)[:5]
        raise ValueError(
            f"Error JSONL contains {len(missing_ids)} study IDs missing from input CSV: "
            f"{preview}"
        )
    completed = completed_study_ids(output_path, expected_inference_settings)
    pending_ids = failed_set.difference(completed)
    pending = [report for report in reports if report.study_id in pending_ids]
    return pending, len(failed_ids), len(failed_set.intersection(completed))


def retry_error_output_path(error_path: Path) -> Path:
    """Return a retry error path that cannot overwrite the source audit log."""
    if error_path.suffix == ".jsonl":
        return error_path.with_name(f"{error_path.stem}.retry.jsonl")
    return error_path.with_name(f"{error_path.name}.retry.jsonl")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--error-output", type=Path)
    parser.add_argument("--metrics-output", type=Path)
    parser.add_argument(
        "--retry-errors-from",
        type=Path,
        help=(
            "retry only study IDs in this error JSONL, excluding IDs already "
            "present in the main output"
        ),
    )
    parser.add_argument("--id-col", default="StudyInstanceUID")
    parser.add_argument("--report-col", default="report")
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT_PATH)
    parser.add_argument("--prompt-version", default=DEFAULT_OPENAI_PROMPT_VERSION)
    parser.add_argument("--model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument(
        "--reasoning-effort",
        choices=REASONING_EFFORTS,
        default="medium",
    )
    parser.add_argument("--max-output-tokens", type=int, default=4_096)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout-seconds", type=float, default=180.0)
    parser.add_argument("--max-retries", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = build_parser().parse_args(argv)
    settings = OpenAIInferenceSettings(
        model=args.model,
        reasoning_effort=args.reasoning_effort,
        max_output_tokens=args.max_output_tokens,
        workers=args.workers,
        timeout_seconds=args.timeout_seconds,
        max_retries=args.max_retries,
    )
    if settings.workers < 1:
        raise ValueError("workers must be positive")
    if settings.max_output_tokens < 1:
        raise ValueError("max output tokens must be positive")
    if args.retry_errors_from is not None and not args.resume:
        raise ValueError("Error retry appends to output; specify --resume")

    prompt = args.prompt.read_text(encoding="utf-8").strip()
    reports = read_reports(args.input, args.id_col, args.report_col)
    if args.retry_errors_from is not None:
        if not args.output.is_file():
            raise FileNotFoundError(
                f"Main output does not exist for error retry: {args.output}"
            )
        pending, retry_requested, retry_already_completed = select_retry_reports(
            reports,
            args.retry_errors_from,
            args.output,
            asdict(settings),
        )
        completed_count = retry_already_completed
        LOGGER.info(
            "Loaded %d reports; retry requested for %d errors, %d already "
            "complete and %d pending",
            len(reports),
            retry_requested,
            retry_already_completed,
            len(pending),
        )
    else:
        completed = (
            completed_study_ids(args.output, asdict(settings)) if args.resume else set()
        )
        if args.output.exists() and not args.resume and not args.dry_run:
            raise FileExistsError(f"Output already exists; use --resume: {args.output}")
        pending = [report for report in reports if report.study_id not in completed]
        completed_count = len(completed)
        retry_requested = 0
        retry_already_completed = 0
        LOGGER.info(
            "Loaded %d reports; %d complete and %d pending",
            len(reports),
            len(completed),
            len(pending),
        )
    if args.dry_run:
        return 0

    if args.retry_errors_from is not None:
        error_path = args.error_output or retry_error_output_path(
            args.retry_errors_from
        )
        if error_path.resolve() == args.retry_errors_from.resolve():
            raise ValueError("Retry error output must differ from the source error JSONL")
        metrics_path = args.metrics_output or args.output.with_suffix(
            ".retry.metrics.json"
        )
    else:
        error_path = args.error_output or args.output.with_suffix(".errors.jsonl")
        metrics_path = args.metrics_output or args.output.with_suffix(".metrics.json")
    metrics = run_inference(
        pending,
        args.output,
        error_path,
        prompt,
        args.prompt_version,
        settings,
    )
    metrics.update(
        {
            "prompt_version": args.prompt_version,
            "model": settings.model,
            "inference_settings": asdict(settings),
            "schema_version": OPENAI_SCHEMA_VERSION,
            "resumed_studies": completed_count,
            "retry_source": (
                str(args.retry_errors_from)
                if args.retry_errors_from is not None
                else None
            ),
            "retry_requested_studies": retry_requested,
            "retry_already_completed_studies": retry_already_completed,
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
