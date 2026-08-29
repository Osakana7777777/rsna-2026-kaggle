#!/usr/bin/env python3
"""Generate reproducible EDA tables, charts, and DICOM image panels.

Run in Kaggle:
    python scripts/run_eda.py --output-dir /kaggle/working/outputs/eda
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rsna2026.eda import main


if __name__ == "__main__":
    raise SystemExit(main())
