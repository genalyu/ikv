#!/usr/bin/env python3
"""Task analysis, preparation, preflight and training."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from n0_twam.task_pipeline.cli import main

if __name__ == "__main__":
    main()
