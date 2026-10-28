import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import quadrl  # noqa: E402,F401  (puts plants/ on cf's search path)
