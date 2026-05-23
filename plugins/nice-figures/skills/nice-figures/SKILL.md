---
name: nice-figures
description: "Generate matplotlib figures in the soft-pastel + warm-earth visual register used by AI alignment research blogs (Anthropic-style) — bold sans-serif display titles, scatter overlaid with smoothed trends and shaded confidence bands, minimal axes, '↓better' badges, warm coral/peach/sage/blue palettes, white background by default. Use this skill whenever the user asks for plots in an Anthropic-style, alignment-research style, soft-pastel style, research-blog style, or conference-poster style. Trigger especially for training-curve plots over RL steps, agentic misalignment bar charts, eval comparisons, scatter plots with error bars and baseline bands, or any figure the user wants to feel more 'research-blog' than 'Nature column.' Also trigger when the user references a previous figure built in this style or asks for conference-ready figures with a clean white background."
---

# Soft Research Figures

A matplotlib visual register for AI research figures — alignment plots,
training curves, eval bar charts, scaling scatter plots. Bold sans-serif
titles, soft pastel + warm earth palettes, smoothed trends with shaded
bands, minimal axes. Conference-ready (white background) by default.

Use this skill for posters, slide decks, full-page appendix figures, blog
posts, and conferences that lean toward modern ML aesthetics (NeurIPS, ICML,
ICLR talks/posters). It favors a relaxed, presentation-friendly look rather
than the tight single-column widths of a strict print journal — but the
white-background default is clean enough for most paper figures too.

## First steps

1. Copy `scripts/soft_style.py` to the working directory.
2. Read `references/chart_recipes.md` and pick the archetype closest to
   what the user described.
3. Plan: how many panels, what data, what colors, which "better" direction.
4. Write the script using the recipe; adapt only what's necessary.
5. Export both PDF and PNG via `save_figure()`.

## When to use this skill

Trigger on phrases like:
- "alignment plot," "training curve," "eval comparison"
- "Anthropic-style figure," "research-blog style," "soft pastel"
- "conference figure," "poster figure," "talk slide chart"
- "make me a [bar chart / ROC / heatmap / box plot / parity plot / scaling law / distribution] in our soft style"
- "match the figures I showed you" (when reference images are present)
- "make it pretty for the appendix"

Don't trigger on:
- Pure data exploration (Jupyter scratch plots) — overkill
- Schematics, diagrams, flowcharts — wrong tool entirely

## The sixteen archetypes

**Anthropic-blog originals (1–5):** the soft-pastel research-figure
register, smoothing bands, rounded bars.

| # | Recipe | Use for |
|---|--------|---------|
| 1 | Multi-panel trend with band | Training curves, multi-benchmark comparison |
| 2 | Scatter with error bars + baseline | Scaling plots, ablation comparisons |
| 3 | Grouped bars + error bars (multi-condition) | Model comparison across metrics |
| 4 | Grouped bars (3 conditions × N scenarios) | Intervention comparison |
| 5 | Multi-line with markers | Single-parameter sweeps |

**Standard publication (6–11):** ROC, heatmaps, distributions, etc.

| # | Recipe | Use for |
|---|--------|---------|
| 6 | Heatmap / confusion matrix | Similarity matrices, attention, ablation grids |
| 7 | ROC / PR curve | Classification diagnostics |
| 8 | Distribution comparison (hist + KDE) | Property distributions, score shifts |
| 9 | Box / violin plot | Seed stability, dataset comparison |
| 10 | Scaling-law plot | Log-log scatter with power-law fit |
| 11 | Parity / calibration plot | Predicted vs. actual, regression diagnostics |

**Domain-specific (12–16):** embedding scatter, ECDFs, forest plots,
horizontal rankings, Pareto trade-offs.

| # | Recipe | Use for |
|---|--------|---------|
| 12 | 2D embedding scatter | t-SNE / UMAP / PCA visualizations |
| 13 | ECDF / CDF plot | Distribution comparison without binning, tail behavior |
| 14 | Forest plot / dot-and-whisker | Effect sizes with CIs across many interventions |
| 15 | Horizontal bar chart (sorted) | Many-method comparisons (>8 categories) |
| 16 | Pareto front / trade-off plot | Accuracy-vs-cost, quality-vs-latency trade-offs |

Full code for each lives in `references/chart_recipes.md`. Copy the
nearest one and adapt.

## Style at a glance

```python
from soft_style import (
    configure_style, figure_title, panel_subtitle, better_badge,
    smooth_curve, rolling_band, rounded_bars, rounded_hbars, save_figure,
    LINE_PALETTE, BAR_PALETTE, MULTILINE_PALETTE,
    CMAP_SEQUENTIAL, CMAP_DIVERGING,
)

configure_style()           # white background (default, conference-ready)
# configure_style(cream_bg=True)  # original warm cream blog background
```

### Palettes

- `LINE_PALETTE` — `blue`, `mustard`, `sage`, `pink`. Use for line plots
  with shaded bands. Always assign colors in this order for visual
  consistency across figures in the same paper.
- `BAR_PALETTE` — `coral`, `peach`, `dark_gray`, `light`. Warm-earth
  grouped bar charts. The coral and peach pair particularly well to show
  "baseline vs. intervention."
- `MULTILINE_PALETTE` — `teal`, `pink`, `dark_green`, `mustard`, `blue`.
  Up to 5 categorical lines. Use white-edged markers on every point.
- `CMAP_SEQUENTIAL` — white → cream → peach → coral → dark coral.
  Use for unsigned data: similarity matrices, confusion matrices,
  attention weights, density heatmaps.
- `CMAP_DIVERGING` — blue → cream → coral. Use for centered data with
  symmetric `vmin`/`vmax`: correlation matrices, log-fold-change,
  signed deviations from a baseline.

### Typography

- Figure title: 18 pt, bold, near-black (`figure_title()`)
- Panel subtitle: 12 pt, normal, gray (`panel_subtitle()`)
- Axis labels: 11 pt, gray (`#6B6960`) — set automatically
- Tick labels: 9.5 pt, lighter gray (`#8C8A82`) — set automatically
- Legend: 9.5 pt, white-filled rounded box with thin gray edge

Inter is the preferred font; the skill falls back to Helvetica, Arial,
then DejaVu Sans. On systems with Inter installed, figures pick it up
automatically.

### Axes

- Only bottom and left spines, in soft gray (`#B8B5AC`)
- No grid lines
- Tick direction outward, short
- Labels in muted gray, never pure black

## Rounded bars (the signature look)

The Anthropic alignment-figure bar charts have a subtle but distinctive
feature: bars sit square on the baseline but the top corners are softly
rounded. Use `rounded_bars()` instead of `ax.bar()`:

```python
from matplotlib.patches import Patch

fig, ax = plt.subplots(figsize=(11, 5.5))
# IMPORTANT: set ylim BEFORE rounded_bars so corners are circular in
# display space — not afterwards
ax.set_xlim(-0.5, n_categories - 0.5)
ax.set_ylim(0, y_max)

for i, (col, lab) in enumerate(zip(colors, conditions)):
    offset = (i - (n - 1) / 2) * width
    rounded_bars(ax, x + offset, heights, width=width, color=col,
                 radius_frac=0.20, label=lab)
    ax.errorbar(x + offset, heights, yerr=errs, fmt="none",
                ecolor="#3a3a37", elinewidth=1.1, capsize=3, zorder=3)

# Use Patch handles for the legend — PathPatch auto-handles look wrong
handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, conditions)]
ax.legend(handles=handles, loc="upper right")
```

`radius_frac` is corner radius as a fraction of bar width:
- 0.10–0.15: barely rounded (subtle)
- **0.18–0.22: matches the Anthropic look (default sweet spot)**
- 0.30–0.40: pronounced rounded caps
- 0.50: half-pill cap (fully rounded top)

The corner radius is computed in display pixels then converted back to
data units, so corners look consistently circular regardless of axis
aspect ratio. This requires axis limits to be set *before* calling
`rounded_bars()`. Forgetting this is the most common gotcha — corners
will look stretched if you set `ylim` after.

For very short bars (height < radius), the corner caps at half the
height, giving a nice rounded-button look. This is what you want.

**Horizontal bars use `rounded_hbars()`** with the same conventions
mirrored 90°: bars grow from x=0 to the right, with a square left
edge (flush against the baseline) and softly rounded right corners.
The signature is `rounded_hbars(ax, y, widths, height, color, radius_frac=0.20, ...)`.
Use this for sorted leaderboards, method rankings, and any other
many-category comparison (recipe 15).

## Background choice

The skill defaults to **white** because that's what conferences and
journals want. The original Anthropic blog figures use a warm cream
outer + lighter cream panel — opt in with `configure_style(cream_bg=True)`
only when generating blog-style assets, never for paper submissions.

When in doubt: white.

## Smoothing and confidence bands

For training-trace data:

```python
# Smoothed trend line
xs, ys = smooth_curve(steps, scores, frac=0.30)
ax.plot(xs, ys, color=c, linewidth=2.0)

# Shaded ± 0.85·σ band (visually tight, not statistically formal)
xs, lo, hi, _ = rolling_band(steps, scores, frac=0.30, k=0.85)
ax.fill_between(xs, lo, hi, color=c, alpha=0.18, linewidth=0)

# Underlying scatter
ax.scatter(steps, scores, color=c, s=8, alpha=0.55, linewidth=0)
```

`frac` controls smoothing bandwidth (fraction of points in the kernel).
**0.30 is the sweet spot for typical training curves.** Drop to 0.15–0.20
if you want to preserve sharp transitions; push up to 0.40+ for very
noisy data. `k=0.85` gives a visually tight band that reads as
"trend uncertainty"; raise to k=1.0 for ±1σ or k=1.96 for ~95% CI.

For real LOWESS with handling for outliers and non-uniform x, prefer
`statsmodels.nonparametric.smoothers_lowess` — `soft_style`'s smoother
is a lightweight Gaussian-kernel approximation that depends only on
numpy.

## "↓better" / "↑better" badge

Place a small rounded badge in any corner of an axis to make the
"direction of better" unambiguous:

```python
better_badge(ax, direction="down", loc="upper right")
```

Skip this badge when:
- The metric is already obviously directional (accuracy, F1)
- The plot has only one series (the trend is its own narrative)
- The axis label already says "(lower=better)" or "(↑ higher is better)"

Use it when:
- Multiple panels share the same y-axis semantics
- The metric is unusual or ambiguous (e.g., "misalignment score")
- A reader skimming the figure should not have to read the caption

## Common mistakes

1. **Mixing palettes.** Don't combine `LINE_PALETTE` and `MULTILINE_PALETTE`
   in the same figure; their blues and pinks differ slightly and look
   like printing errors. Pick one palette per figure.
2. **Too many series per panel.** This register handles ~4 series
   cleanly. Beyond 5, split into small multiples.
3. **Forgetting `figure_title()`.** The bold display title is a defining
   feature of this register. Don't substitute `fig.suptitle(text)` with
   default weight — it looks like a different style.
4. **Sharing y-axis when scales differ.** In the multi-panel archetype,
   independent y-scales communicate "these are separate benchmarks." Only
   share when the panels are directly comparable.
5. **Cream background for paper figures.** Reviewers will flag it as
   off-spec for nearly every conference. White by default; cream only
   for blog/marketing assets.
6. **Pure-black text.** Use `#1B1A18` for titles, `#6B6960` for labels,
   `#8C8A82` for ticks. The skill sets these automatically — don't
   override with `color="black"`.
7. **Using `ax.bar()` instead of `rounded_bars()`.** Square-topped bars
   read as "different aesthetic" — they don't belong in this register.
   Always use `rounded_bars()` for bar charts.
8. **Setting `ylim` after `rounded_bars()`.** Corner radii are computed
   at call time using current axis limits. Always set limits first.

## Export

```python
save_figure(fig, "figure_1")
# Saves figure_1.pdf and figure_1.png at 300 dpi with bbox_inches="tight"
# and facecolor matching the figure (white or cream)
```

For very large figures or print-quality posters:

```python
save_figure(fig, "poster_fig", formats=("pdf", "png"), dpi=600)
```

## Checklist before delivering

- [ ] `configure_style()` called *before* any plotting
- [ ] Title applied via `figure_title()` (bold, large, sentence case)
- [ ] Each panel labeled via `panel_subtitle()` (multi-panel only)
- [ ] Y-axis units / direction-of-better clear (badge or label suffix)
- [ ] Confidence band, smoothing, and scatter all in same color per series
- [ ] No grid, only bottom + left spines
- [ ] White background unless `cream_bg=True` was explicitly requested
- [ ] Exported as both PDF and PNG via `save_figure()`
- [ ] If used in a paper: figure also legible at 50% scale (sanity check)

## If the user has actual data

When real arrays or a CSV are provided:
1. Load the data first; print shapes and a head so the user can confirm.
2. Map columns to the recipe's variables. Don't invent placeholder data
   alongside theirs — replace synthetic data entirely.
3. If multiple seeds are present, prefer `fill_between(mean ± std)` over
   `rolling_band()`, which uses kernel smoothing instead.

When the user describes a figure conceptually with no data, generate
realistic synthetic data via `np.random.seed(...)` for reproducibility,
and clearly mark it as a placeholder in a comment.
