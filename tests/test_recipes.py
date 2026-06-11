"""Every ```python block in chart_recipes.md must actually run.

Recipes are copy-and-adapt material for an agent: a block that NameErrors
or has drifted from soft_style's API is worse than no recipe at all. Each
`## section` is executed in a fresh namespace seeded with the shared
preamble (the md's own "All recipes assume" block) plus synthetic stand-ins
for the user-supplied variables some recipes reference. Blocks within a
section run sequentially, since variant blocks build on the main block.
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
RECIPES_MD = (REPO / "plugins/nice-figures/skills/nice-figures"
              / "references/chart_recipes.md")

BLOCK_RE = re.compile(r"```python\n(.*?)```", re.S)


def _sections():
    """Yield (title, [code blocks]) for every section that has code."""
    text = RECIPES_MD.read_text()
    # First python block of the file is the shared preamble
    preamble = BLOCK_RE.search(text).group(1)
    parts = text.split("\n## ")
    out = []
    for part in parts[1:]:
        title = part.split("\n", 1)[0].strip()
        blocks = BLOCK_RE.findall(part)
        if blocks:
            out.append((title, blocks))
    return preamble, out


PREAMBLE, SECTIONS = _sections()


def _fixtures():
    """Synthetic stand-ins for variables recipes expect from the user."""
    import soft_style as ss
    rng = np.random.default_rng(0)

    fpr = np.linspace(0, 1, 50)
    seeds = lambda mu: rng.normal(mu, 0.05, 12)  # noqa: E731
    y_true = rng.normal(1.0, 1.2, 200)

    fx = {
        # recipe 6: heatmap
        "labels": ["alpha", "beta", "gamma", "delta", "epsilon"],
        # recipe 7: ROC
        "methods": {
            "Model A": {"fpr": fpr, "tpr": fpr ** 0.4, "auc": 0.86,
                        "color": ss.LINE_PALETTE["blue"]},
            "Model B": {"fpr": fpr, "tpr": fpr ** 0.7, "auc": 0.71,
                        "color": ss.LINE_PALETTE["pink"]},
        },
        # recipe 8: distributions
        "baseline_data": rng.beta(4, 3, 800),
        "molcross_data": rng.beta(5, 2.5, 800),
        "reinvent_data": rng.beta(3, 4, 800),
        # recipes 9 + 13: per-method losses
        "adamw_losses": seeds(2.31),
        "lion_losses": seeds(2.28),
        "muon_losses": seeds(2.22),
        "adaptmuon_losses": seeds(2.19),
        "adaptive_muon_losses": seeds(2.19),
        # recipe 10 variant: final losses + precomputed fit
        "compute": np.logspace(16, 20, 6),
        "loss_obs": 100.0 * np.logspace(16, 20, 6) ** -0.05,
        "C0": 100.0,
        "alpha": 0.05,
        "c": ss.LINE_PALETTE["blue"],
        # recipe 11: parity
        "y_true": y_true,
        "y_pred": y_true + rng.normal(0, 0.4, 200),
    }
    # recipe 6: row-normalized confusion matrix
    m = rng.random((5, 5)) + 4 * np.eye(5)
    fx["matrix"] = m / m.sum(axis=1, keepdims=True)
    return fx


@pytest.mark.parametrize(
    "title,blocks", SECTIONS, ids=[t[:40] for t, _ in SECTIONS])
def test_recipe_section_runs(title, blocks, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # save_figure() writes here
    ns = {}
    exec(compile(PREAMBLE, "<preamble>", "exec"), ns)
    ns["np"].random.seed(0)
    ns.update(_fixtures())
    try:
        for i, block in enumerate(blocks):
            exec(compile(block, f"<{title} block {i}>", "exec"), ns)
    finally:
        plt.close("all")
