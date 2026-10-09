"""Lets every module's tests import the shared helpers in code_checks.py."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
