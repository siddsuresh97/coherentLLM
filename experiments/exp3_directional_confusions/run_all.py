#!/usr/bin/env python3
"""Convenience wrapper for the Experiment 3 Step 1 pipeline."""

from pathlib import Path
import runpy
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.argv = [str(ROOT / "scripts" / "run_experiment3.py"), "all", *sys.argv[1:]]
runpy.run_path(str(ROOT / "scripts" / "run_experiment3.py"), run_name="__main__")
