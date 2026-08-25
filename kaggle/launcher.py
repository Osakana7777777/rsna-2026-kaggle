"""Kaggle launcher with project files embedded by scripts/prepare.sh."""

from __future__ import annotations

import contextlib
import shutil
import sys
import traceback
from pathlib import Path
from typing import TextIO


EMBEDDED_FILES: dict[str, str] = {}
RUN_MODE = "__RUN_MODE__"


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

from rsna2026.train import main  # noqa: E402


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


if __name__ == "__main__":
    output_dir = Path("/kaggle/working/outputs")
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "run.log").open("w", encoding="utf-8") as log_file:
        with contextlib.redirect_stdout(Tee(sys.stdout, log_file)):
            with contextlib.redirect_stderr(Tee(sys.stderr, log_file)):
                try:
                    exit_code = main(
                        [
                            "--config",
                            str(PAYLOAD_ROOT / "configs" / "baseline.toml"),
                            "--mode",
                            RUN_MODE,
                        ]
                    )
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
