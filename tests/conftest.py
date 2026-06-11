"""Shared test setup: headless matplotlib + soft_style on the path."""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "plugins/nice-figures/skills/nice-figures/scripts"
sys.path.insert(0, str(SCRIPTS))
