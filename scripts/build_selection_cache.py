"""Build the pinned, compressed slice-selection cache for Kaggle."""

from __future__ import annotations

import argparse
import hashlib
import lzma
import os
from pathlib import Path
from typing import Sequence


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    raw = args.source.read_bytes()
    actual_sha256 = hashlib.sha256(raw).hexdigest()
    if actual_sha256 != args.expected_sha256:
        raise ValueError(f"Selection cache SHA-256 mismatch: {actual_sha256}")

    compressed = lzma.compress(raw, format=lzma.FORMAT_XZ, preset=6)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".temporary")
    temporary.write_bytes(compressed)
    os.replace(temporary, args.output)
    print(
        f"Built {args.output}: raw={len(raw)}, compressed={len(compressed)}, "
        f"raw_sha256={actual_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
