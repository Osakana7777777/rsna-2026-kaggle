"""Convert canonical weak-label JSONL into an analysis Parquet table."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from rsna2026.weak_label_decode import flatten_prediction


def load_flat_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                study_id = str(record["study_id"])
                row = flatten_prediction(study_id, record["prediction"])
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"Invalid JSONL line {line_number}: {error}"
                ) from error
            if study_id in seen:
                raise ValueError(f"Duplicate study identifier: {study_id}")
            seen.add(study_id)
            row["prompt_version"] = str(record.get("prompt_version", ""))
            row["model"] = str(record.get("model", ""))
            rows.append(row)
    return rows


def write_parquet(rows: list[dict[str, object]], path: Path) -> None:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as error:
        raise RuntimeError(
            "pyarrow is required to create the Parquet output"
        ) from error
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    metadata = dict(table.schema.metadata or {})
    metadata[b"canonical_source"] = b"raw_predictions.jsonl"
    metadata[b"format_version"] = b"rsna-knee-extractor-v1"
    table = table.replace_schema_metadata(metadata)
    pq.write_table(table, path, compression="zstd")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    rows = load_flat_rows(args.input)
    if not rows:
        raise ValueError("Input contains no predictions")
    write_parquet(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
