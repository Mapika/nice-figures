# nice-figures — instructions for AI agents

This repo provides a matplotlib visual register for scientific and
research-blog figures: bold display titles, soft pastel palettes,
smoothed trends with shaded bands, signature rounded bars, minimal
axes. White background by default — output is conference- and
paper-ready (PDFs embed Type 42 fonts).

If you are an agent asked to produce a plot in this style — or just a
*good-looking* research figure — follow this file. It is the
agent-agnostic equivalent of the Claude Code skill in
`plugins/nice-figures/skills/nice-figures/SKILL.md`.

## Get the style helper

Either of:

```bash
pip install git+https://github.com/Mapika/nice-figures   # import soft_style
# or copy the single file (matplotlib + numpy only):
curl -O https://raw.githubusercontent.com/Mapika/nice-figures/main/plugins/nice-figures/skills/nice-figures/scripts/soft_style.py
```

## Workflow

1. Pick the closest archetype from
   [`chart_recipes.md`](plugins/nice-figures/skills/nice-figures/references/chart_recipes.md)
   — 16 tested, copy-paste-runnable recipes (trend bands, rounded bars,
   heatmaps, ROC, distributions, scaling laws, forest plots, Pareto
   fronts, …). Copy it and adapt only what's necessary.
2. Call `configure_style()` **before** any plotting
   (`scale=0.75` for single-column paper figures).
3. Export with `save_figure(fig, name)` → PDF + PNG at 300 dpi.
4. **Render, then look at the PNG with your vision capability before
   delivering.** Check: legend not covering data (`top_legend()` exists
   for this), every multi-series plot has a legend or colorbar, nothing
   clipped, tick labels not colliding. Fix and re-render until clean.
   This step catches what running the code cannot.

## Hard rules of the register

- Bar charts use `rounded_bars()` / `rounded_hbars()`, never plain
  `ax.bar()` — and axis limits are set **before** calling them.
- Titles via `figure_title()` (bold), panel labels via `panel_subtitle()`.
- One palette per figure (`LINE_PALETTE`, `BAR_PALETTE`,
  `MULTILINE_PALETTE`); `CMAP_GRADIENT` + colorbar for 6+ ordered series.
- No grid; bottom + left spines only; no pure-black text
  (use `NEUTRAL["ink"]` for error bars and emphasis lines).
- White background unless the user explicitly wants the cream blog look.

Full conventions, common mistakes, and the pre-delivery checklist:
[`SKILL.md`](plugins/nice-figures/skills/nice-figures/SKILL.md).

## Minimal example

```python
import numpy as np, matplotlib.pyplot as plt
from soft_style import (configure_style, figure_title, rounded_bars,
                        save_figure, BAR_PALETTE, NEUTRAL, top_legend)
from matplotlib.patches import Patch

configure_style()
fig, ax = plt.subplots(figsize=(9, 5))
ax.set_xlim(-0.5, 2.5); ax.set_ylim(0, 1.0)          # limits FIRST
rounded_bars(ax, np.arange(3), [0.82, 0.64, 0.91], width=0.55,
             color=BAR_PALETTE["coral"])
ax.set_xticks(range(3)); ax.set_xticklabels(["A", "B", "C"])
ax.set_ylabel("Accuracy")
figure_title(fig, "Headline result", y=1.0)
save_figure(fig, "figure_1")                          # PDF + PNG
# Now READ figure_1.png and inspect it before delivering.
```
