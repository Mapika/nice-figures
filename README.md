# nice-figures

A Claude Code plugin that teaches Claude to generate **matplotlib figures in a soft-pastel, research-blog visual register** — the kind of plot you see in modern ML/alignment write-ups. Bold sans-serif display titles, scatter overlaid with smoothed trends and shaded confidence bands, signature rounded bars, minimal axes, and `↓better` badges. White background by default, so the output is conference- and paper-ready.

It ships as a single skill (`nice-figures`) plus a dependency-light style helper and 16 ready-to-adapt chart recipes. Claude invokes it automatically when you ask for plots in this style, or you can call it explicitly.

## Install

This repo is a Claude Code plugin marketplace. Add it, then install the plugin:

```
/plugin marketplace add Mapika/nice-figures
/plugin install nice-figures@nice-figures
```

To update later:

```
/plugin marketplace update nice-figures
```

## Usage

Once installed, just describe the figure you want. Claude triggers the skill on requests like:

- "Make a training-curve plot of these RL scores with a smoothed trend and a shaded band, soft research-blog style."
- "Grouped bar chart comparing three models across four eval scenarios, with the rounded bar tops and error bars."
- "Scaling-law scatter on log-log axes with a power-law fit for our poster — clean white background."
- "Match the figure you made last week with the coral/peach bars."

You can also invoke it directly:

```
/nice-figures:nice-figures
```

Bring your own data (CSV or arrays) and Claude maps it onto the nearest recipe; describe a figure with no data and it generates a clearly-marked synthetic placeholder.

## What's inside

```
plugins/nice-figures/skills/nice-figures/
├── SKILL.md                     # the skill: when to use it, conventions, gotchas
├── scripts/soft_style.py        # matplotlib style helpers (numpy + matplotlib only)
└── references/chart_recipes.md  # full code for all 16 chart archetypes
```

### The style

- **Palettes** — `LINE_PALETTE` (blue/mustard/sage/pink) for trend lines, `BAR_PALETTE` (coral/peach/gray/olive) for bars, `MULTILINE_PALETTE` for up to 5 categorical lines, plus sequential and diverging colormaps.
- **Rounded bars** — the signature look. `rounded_bars()` / `rounded_hbars()` give softly rounded top corners computed in display space so they stay circular at any aspect ratio.
- **Smoothing + bands** — `smooth_curve()` and `rolling_band()` for trend lines with tight shaded uncertainty bands.
- **Typography & axes** — bold display title, gray subtitles and labels (never pure black), only bottom/left spines, no grid.
- **Export** — `save_figure()` writes both PDF and PNG at 300 dpi.

### The 16 archetypes

Trend-with-band, scatter + baseline, grouped bars, multi-line sweeps, heatmaps/confusion matrices, ROC/PR curves, distribution comparisons, box/violin, scaling-law fits, parity/calibration, 2D embeddings, ECDFs, forest plots, sorted horizontal bars, and Pareto fronts. Full copy-and-adapt code lives in `references/chart_recipes.md`.

## Requirements

- `matplotlib` and `numpy` (the only hard dependencies of the style helper).
- The [Inter](https://rsms.me/inter/) font is preferred and picked up automatically if installed; otherwise it falls back to Helvetica → Arial → DejaVu Sans.

## License

MIT © Mark Marosi
