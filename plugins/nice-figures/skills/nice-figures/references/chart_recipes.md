# Chart Recipes

Eleven archetypes, each matching a common publication plot type, all in
the same soft-research visual register. Copy the recipe most appropriate
to your data and adapt.

## Contents

**Anthropic-blog archetypes (1–5):**
1. [Multi-panel trend with scatter + smoothed line + shaded band](#1-multi-panel-trend-with-scatter--smoothed-line--shaded-band)
2. [Scatter with error bars + baseline band (log-x)](#2-scatter-with-error-bars--baseline-band-log-x)
3. [Grouped bar chart with error bars (warm-earth palette)](#3-grouped-bar-chart-with-error-bars-warm-earth-palette)
4. [Grouped bars with three conditions (Image 4 variant)](#4-grouped-bars-with-three-conditions-image-4-variant)
5. [Multi-line plot with categorical legend](#5-multi-line-plot-with-categorical-legend)

**Cross-archetype tips** ([jump](#cross-archetype-tips))

**Standard publication archetypes (6–11):**

6. [Heatmap / confusion matrix](#6-heatmap--confusion-matrix)
7. [ROC / PR curves](#7-roc--pr-curves)
8. [Distribution comparison (histograms + KDE)](#8-distribution-comparison-histograms--kde)
9. [Box / violin plot with overlaid points](#9-box--violin-plot-with-overlaid-points)
10. [Scaling-law plot (per-run curves + compute-efficient frontier)](#10-scaling-law-plot-per-run-curves--compute-efficient-frontier)
11. [Parity / calibration plot](#11-parity--calibration-plot)

**Domain-specific archetypes (12–16):**

12. [2D embedding scatter (t-SNE / UMAP / PCA)](#12-2d-embedding-scatter-t-sne--umap--pca)
13. [ECDF / CDF plot](#13-ecdf--cdf-plot)
14. [Forest plot / dot-and-whisker](#14-forest-plot--dot-and-whisker)
15. [Horizontal bar chart (sorted ranking)](#15-horizontal-bar-chart-sorted-ranking)
16. [Pareto front / trade-off plot](#16-pareto-front--trade-off-plot)

All recipes assume:

```python
import matplotlib.pyplot as plt
import numpy as np
from soft_style import (
    configure_style, figure_title, panel_subtitle, better_badge,
    top_legend, plain_log_ticks, soft_colorbar,
    smooth_curve, rolling_band, rounded_bars, rounded_hbars, save_figure,
    LINE_PALETTE, BAR_PALETTE, MULTILINE_PALETTE,
    CMAP_SEQUENTIAL, CMAP_DIVERGING, CMAP_GRADIENT, NEUTRAL,
)

configure_style()  # white background by default
```

---

## 1. Multi-panel trend with scatter + smoothed line + shaded band

The "Rate of alignment failures over RL steps" archetype. Three panels
side by side, each showing several noisy training-trace series. Each
series is rendered as: light-translucent shaded band (mean ± σ), a
smoothed trend line, and the underlying scatter points.

```python
np.random.seed(0)
steps = np.linspace(0, 600, 80)
panels = ["Blackmail", "Financial crimes", "Cancer research"]

# Synthetic data: four series per panel, each a noisy decay
def make_series(start, decay):
    base = start * np.exp(-steps / decay)
    return base + np.random.normal(0, start * 0.15, len(steps))

panel_data = {
    "Blackmail": {
        "blue":    make_series(0.04, 800),
        "mustard": make_series(0.10, 400),
        "sage":    make_series(0.17, 1200),
        "pink":    make_series(0.18, 600),
    },
    "Financial crimes": {
        "blue":    make_series(0.02, 300),
        "mustard": make_series(0.03, 200),
        "sage":    make_series(0.07, 200),
        "pink":    make_series(0.10, 250),
    },
    "Cancer research": {
        "blue":    make_series(0.05, 400),
        "mustard": make_series(0.04, 500),
        "sage":    make_series(0.14, 350),
        "pink":    make_series(0.22, 300),
    },
}

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5),
                         gridspec_kw=dict(wspace=0.25))

for ax, panel in zip(axes, panels):
    for color_name, y in panel_data[panel].items():
        c = LINE_PALETTE[color_name]
        # Shaded confidence band
        xs, lo, hi, _ = rolling_band(steps, y, frac=0.30, k=0.85)
        ax.fill_between(xs, lo, hi, color=c, alpha=0.18, linewidth=0)
        # Smoothed trend
        xs2, ys2 = smooth_curve(steps, y, frac=0.30)
        ax.plot(xs2, ys2, color=c, linewidth=2.0)
        # Underlying scatter (small, semi-transparent)
        ax.scatter(steps, y, color=c, s=8, alpha=0.55, linewidth=0)

    panel_subtitle(ax, panel)
    ax.set_ylim(bottom=0)
    better_badge(ax, direction="down", loc="upper right")

axes[0].set_ylabel("Score")
fig.text(0.5, -0.02, "Training steps", ha="center",
         color="#6B6960", fontsize=11)

figure_title(fig, "Rate of alignment failures over RL steps", y=1.04)
save_figure(fig, "fig_rl_failures")
```

Notes:
- `panel_subtitle()` produces the centered gray subtitle above each panel.
- For real data with explicit uncertainty (multiple seeds), substitute
  `fill_between(steps, mean - std, mean + std)` for `rolling_band()`.
- Share `ylim` only across panels that should be directly comparable.
  In the reference image the three panels have *different* y-scales,
  emphasizing that each is its own benchmark.

---

## 2. Scatter with error bars + baseline band (log-x)

The "Alignment eval performance vs. training data size" archetype.
Single panel, error bars on each point, log-scale x-axis, dashed
baseline with a shaded tolerance band.

```python
np.random.seed(1)
# Three categories of run
runs = {
    "Difficult advice": {
        "tokens": [3.0],
        "score": [0.013],
        "err":   [0.012],
        "color": LINE_PALETTE["pink"],
    },
    "PM filtering": {
        "tokens": [40.0],
        "score": [0.172],
        "err":   [0.022],
        "color": "#3B6E4D",
    },
    "System prompt injection": {
        "tokens": [15, 23, 23, 28, 30, 35, 38, 38, 45, 50, 50, 50, 90],
        "score":  [0.132, 0.179, 0.115, 0.092, 0.067, 0.117, 0.089,
                   0.150, 0.065, 0.035, 0.022, 0.018, 0.013],
        "err":    [0.018, 0.022, 0.014, 0.018, 0.014, 0.020, 0.015,
                   0.022, 0.018, 0.012, 0.010, 0.010, 0.008],
        "color":  LINE_PALETTE["blue"],
    },
}
baseline = 0.22
baseline_band = 0.02

fig, ax = plt.subplots(figsize=(10, 5.5))

# Baseline dashed line + shaded tolerance band
ax.axhspan(baseline - baseline_band, baseline + baseline_band,
           color=NEUTRAL["band"], alpha=0.5, zorder=1)
ax.axhline(baseline, linestyle="--", color="#6B6960",
           linewidth=1.2, zorder=2)

# Each category
for name, d in runs.items():
    ax.errorbar(d["tokens"], d["score"], yerr=d["err"],
                fmt="o", color=d["color"], ecolor=d["color"],
                markersize=8, linewidth=0, elinewidth=1.2,
                capsize=3, capthick=1.0, label=name, zorder=3)

ax.set_xscale("log")
ax.set_xlabel("Training tokens (M, log scale)")
ax.set_ylabel("Weighted mean score\n(blackmail, cancer, finance)")
ax.set_ylim(-0.02, 0.27)

# Custom legend with baseline entry
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=d["color"], markersize=8, label=name)
    for name, d in runs.items()
]
handles.append(Line2D([0], [0], linestyle="--", color="#6B6960",
                      linewidth=1.2, label="Sonnet 4 baseline"))
ax.legend(handles=handles, loc="upper right")

figure_title(fig, "Alignment eval performance vs. training data size", y=0.99)
save_figure(fig, "fig_eval_vs_tokens")
```

Notes:
- The shaded band around the baseline uses `axhspan` with a light gray.
- A custom legend (`Line2D` handles) is necessary because matplotlib's
  default `errorbar` legend handles include the error-bar caps.
- For very few points in a category (n=1), include the point but note in
  caption that error bars are seed std, not sample CI.

---

## 3. Grouped bar chart with error bars (warm-earth palette)

The "Automated auditing scores by model" archetype. Categories on x-axis,
each category has several bars side by side, error bars on top. Bars
have rounded top corners (the Anthropic signature look).

```python
from matplotlib.patches import Patch

categories = ["Spontaneous\nself preservation", "Misaligned\nbehavior",
              "Deceptive to user", "Sycophancy", "Leaking"]
models = ["Claude Sonnet 4", "Difficult advice (3M tokens)",
          "Synthetic honeypots (23M tokens)",
          "Synthetic honeypots (85M tokens)"]
colors = [BAR_PALETTE["coral"], BAR_PALETTE["peach"],
          BAR_PALETTE["dark_gray"], BAR_PALETTE["light"]]

means = np.array([
    [0.005, 0.226, 0.027, 0.057, 0.003],
    [0.006, 0.172, 0.013, 0.038, 0.009],
    [0.013, 0.226, 0.024, 0.050, 0.018],
    [0.006, 0.215, 0.018, 0.046, 0.011],
])
errs = np.array([
    [0.004, 0.035, 0.014, 0.018, 0.005],
    [0.003, 0.029, 0.008, 0.015, 0.006],
    [0.007, 0.035, 0.015, 0.016, 0.011],
    [0.004, 0.039, 0.012, 0.016, 0.009],
])

fig, ax = plt.subplots(figsize=(11, 5.5))
# IMPORTANT: set limits BEFORE rounded_bars() so corners are circular in
# display space. Setting limits afterwards stretches the corners.
ax.set_xlim(-0.5, len(categories) - 0.5)
ax.set_ylim(0, 0.28)

n_models = len(models)
width = 0.18
x = np.arange(n_cats := len(categories))

for i, (mean_row, err_row, col, lab) in enumerate(
        zip(means, errs, colors, models)):
    offset = (i - (n_models - 1) / 2) * width
    rounded_bars(ax, x + offset, mean_row, width=width, color=col,
                 radius_frac=0.22, label=lab)
    ax.errorbar(x + offset, mean_row, yerr=err_row, fmt="none",
                ecolor=NEUTRAL["ink"], elinewidth=1.1, capsize=3,
                capthick=1.0, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(categories)
ax.set_ylabel("Mean score")
# PathPatch auto-legend handles don't look right — use Patch instead.
# An in-axes corner works HERE because the tallest bar is far from it;
# if any bar reaches the legend corner, switch to top_legend(ax, handles).
handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, models)]
ax.legend(handles=handles, loc="upper right")

figure_title(fig, "Automated auditing scores by model", y=1.00)
save_figure(fig, "fig_auditing_scores")
```

Notes:
- `width=0.18` with 4 bars per category leaves ~0.28 gap between groups.
  Increase width up to 0.22 to tighten clustering, decrease to spread.
- `radius_frac=0.22` matches the Anthropic look. Push higher (0.40+) for
  more pronounced rounded caps; drop to 0.10 for barely-rounded tops.
- For the "lower=better" version (Image 4), set `ylabel` to
  `"Misalignment score (lower=better)"` and consider adding
  `better_badge(ax, "down")`.
- Use `Patch` handles for the legend, not the patches returned from
  `rounded_bars()` — matplotlib's auto-legend handle for `PathPatch`
  renders as the full path silhouette and looks odd.

---

## 4. Grouped bars with three conditions (Image 4 variant)

Same recipe as above, simplified for 3 conditions × 3 scenarios:

```python
from matplotlib.patches import Patch

scenarios = ["Blackmail", "Financial crimes", "Cancer research"]
conditions = ["Baseline", "Constitutional SDF",
              "Constitutional SDF + stories"]
colors = [BAR_PALETTE["coral"], BAR_PALETTE["peach"],
          BAR_PALETTE["dark_gray"]]

means = np.array([
    [0.650, 0.268, 0.181],
    [0.491, 0.111, 0.039],
    [0.673, 0.111, 0.012],
])
errs = np.array([
    [0.018, 0.018, 0.018],
    [0.012, 0.008, 0.005],
    [0.010, 0.010, 0.005],
])

fig, ax = plt.subplots(figsize=(10, 5.5))
# Set limits BEFORE rounded_bars (see recipe 3 note)
ax.set_xlim(-0.5, len(scenarios) - 0.5)
ax.set_ylim(0, 0.75)

width = 0.25
x = np.arange(len(scenarios))

for i, (col, lab) in enumerate(zip(colors, conditions)):
    offset = (i - 1) * width
    rounded_bars(ax, x + offset, means[:, i], width=width, color=col,
                 radius_frac=0.20, label=lab)
    ax.errorbar(x + offset, means[:, i], yerr=errs[:, i], fmt="none",
                ecolor=NEUTRAL["ink"], elinewidth=1.1, capsize=3,
                capthick=1.0, zorder=3)

ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Misalignment score (lower=better)")
# Tall bars occupy both upper corners → legend goes above the axes
handles = [Patch(facecolor=c, label=l) for c, l in zip(colors, conditions)]
top_legend(ax, handles)

figure_title(fig, "Agentic misalignment evals", y=1.06)
save_figure(fig, "fig_agentic_misalignment")
```

**Variant: stacked rounded bars.** Lower segments are plain rectangles;
only the topmost segment gets the rounded corners, anchored on the
cumulative height via `baseline=`. Heights passed to `rounded_bars()`
are absolute tops, not deltas:

```python
from matplotlib.patches import Patch

scenarios = ["Blackmail", "Financial crimes", "Cancer research"]
parts = [  # bottom-to-top stacking order
    ("Complied",          BAR_PALETTE["coral"], np.array([0.42, 0.30, 0.51])),
    ("Partial / hedged",  BAR_PALETTE["peach"], np.array([0.21, 0.18, 0.14])),
    ("Refused",           BAR_PALETTE["dark_gray"], np.array([0.10, 0.08, 0.07])),
]

fig, ax = plt.subplots(figsize=(9, 5.5))
x = np.arange(len(scenarios))
totals = sum(vals for _, _, vals in parts)
ax.set_xlim(-0.5, len(scenarios) - 0.5)   # limits BEFORE rounded_bars
ax.set_ylim(0, totals.max() * 1.15)

bottom = np.zeros(len(scenarios))
for i, (lab, col, vals) in enumerate(parts):
    if i < len(parts) - 1:                # lower segments: square
        ax.bar(x, vals, width=0.55, bottom=bottom, color=col,
               edgecolor="none", zorder=2)
    else:                                 # top segment: rounded corners
        rounded_bars(ax, x, bottom + vals, width=0.55, color=col,
                     baseline=bottom)
    bottom += vals

ax.set_xticks(x)
ax.set_xticklabels(scenarios)
ax.set_ylabel("Fraction of responses")
handles = [Patch(facecolor=col, label=lab) for lab, col, _ in parts]
top_legend(ax, handles)
figure_title(fig, "Response breakdown by scenario", y=1.06)
save_figure(fig, "fig_stacked_bars")
```

Keep stacks to ≤4 segments and order them largest-at-bottom; beyond
that, a grouped chart or small multiples reads better.

---

## 5. Multi-line plot with categorical legend

The "Misalignment over RL training" archetype. Several lines sharing
x and y; each line a categorical condition. Marker on every data point;
line passes through points without smoothing.

```python
steps_x = [10, 50, 90, 130, 170, 210, 250, 300]

series = {
    "0% chat, 100% agentic":  [0.435, 0.445, 0.412, 0.426, 0.412,
                               0.396, 0.395, 0.406],
    "33% chat, 67% agentic":  [0.448, 0.459, 0.435, 0.420, 0.427,
                               0.398, 0.410, 0.392],
    "50% chat, 50% agentic":  [0.421, 0.450, 0.415, 0.448, 0.441,
                               0.424, 0.408, 0.401],
    "75% chat, 25% agentic":  [0.456, 0.433, 0.428, 0.436, 0.435,
                               0.415, 0.415, 0.415],
    "100% chat, 0% agentic":  [0.440, 0.442, 0.440, 0.440, 0.429,
                               0.424, 0.428, 0.428],
}
color_order = ["teal", "pink", "dark_green", "mustard", "blue"]

fig, ax = plt.subplots(figsize=(11, 5.5))

for (label, ys), color_name in zip(series.items(), color_order):
    c = MULTILINE_PALETTE[color_name]
    ax.plot(steps_x, ys, color=c, linewidth=1.8, marker="o",
            markersize=6, markerfacecolor=c, markeredgecolor="white",
            markeredgewidth=1.2, label=label)

ax.set_xlabel("RL training steps")
ax.set_ylabel("Mean misalignment score (lower=better)")
ax.set_xlim(left=0)
leg = ax.legend(loc="upper right", title="RL environment mix")
leg.get_title().set_color("#6B6960")
leg.get_title().set_fontsize(9.5)

figure_title(fig, "Misalignment over RL training", y=1.00)
save_figure(fig, "fig_misalignment_rl")
```

Notes:
- The thin white outline around each marker (`markeredgecolor="white"`)
  cleanly separates dots from the line when they overlap.
- For visual clarity with many series, prefer 4–6 colors max. Beyond
  that, switch to small multiples (one panel per series).

---

## Cross-archetype tips

**Spacing and figure size.** The Anthropic blog uses ~16:9 wide figures.
For a conference paper with a single-column constraint (~3.5 inches),
use `configure_style(scale=0.75)` with `figsize=(3.5, 2.4)` (or
`scale=0.85` for double-column width) — but recognize that this register
loses some of its character at column scale. The skill is best suited to
slide decks, posters, full-page paper figures, or appendices.

**Color combinations to avoid.** Don't mix `LINE_PALETTE` and
`MULTILINE_PALETTE` in the same figure — they have overlapping hues
that read as "near-misses." Pick one and stay in it.

**Annotation overlays.** The original figures sometimes add small
text annotations (e.g., a callout to a specific data point). Keep
these in `NEUTRAL["label"]` color at `fontsize=9` for consistency.

**Subplot sharing.** If panels show comparable benchmarks (same scale,
same units), pass `sharey=True` to `plt.subplots()`. If the scales
differ meaningfully (as in the reference Image 1), keep them
independent — different y-ranges communicate "different benchmarks."

**Math in titles.** Matplotlib renders math (`$\log P$`, `$R^2$`) in
mathtext, which doesn't inherit the bold weight from `figure_title()`.
Two workarounds: (1) avoid math in titles, write `logP` and `R^2` as
plain text; (2) wrap math in `\mathbf{}`, e.g. `r"$\mathbf{R^2}$ vs n"`.
Axis labels and annotations handle math correctly.

---

# Advanced archetypes (6 – 11)

The following recipes cover the most common scientific plot types
beyond the original five: heatmaps, ROC curves, distributions, box
plots, scaling laws, and parity plots.

---

## 6. Heatmap / confusion matrix

Square cells, custom soft-palette colormap (`CMAP_SEQUENTIAL` for
unsigned data, `CMAP_DIVERGING` for centered data), cell annotations
that flip color on dark cells, minimal axes (no spines).

```python
from soft_style import CMAP_SEQUENTIAL, NEUTRAL

# data: (n, n) array, e.g. row-normalized confusion matrix
fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(matrix, cmap=CMAP_SEQUENTIAL, vmin=0, vmax=1, aspect="equal")

# Annotate cells — flip text color based on cell brightness
for i in range(matrix.shape[0]):
    for j in range(matrix.shape[1]):
        v = matrix[i, j]
        if v > 0.005:  # don't clutter with zeros
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    fontsize=9,
                    color="white" if v > 0.55 else NEUTRAL["title"])

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=35, ha="right")
ax.set_yticklabels(labels)
ax.set_xlabel("Predicted class")
ax.set_ylabel("True class")

# Heatmaps look cleaner without spines or tick marks
for spine in ax.spines.values():
    spine.set_visible(False)
ax.tick_params(length=0)

# Colorbar with soft styling
soft_colorbar(fig, ax, im, label="Row-normalized count",
              fraction=0.04, pad=0.04)

figure_title(fig, "Functional-group classifier confusion matrix", y=1.02)
save_figure(fig, "fig_heatmap")
```

Notes:
- For centered data (e.g., correlation matrices, log-fold-change), use
  `CMAP_DIVERGING` with symmetric `vmin=-x, vmax=+x`.
- The brightness threshold (0.55) for flipping text color works for
  `CMAP_SEQUENTIAL`. For `CMAP_DIVERGING`, flip both at high positive
  AND high negative ends: `color = "white" if abs(v) > 0.55 else ...`.
- For non-square cells (e.g., ablation table), drop `aspect="equal"`.
- Large matrices (>20×20) become unreadable with cell annotations.
  Drop the inner loop and rely on the colorbar instead.

---

## 7. ROC / PR curves

Square axis, multiple curves on the same panel, AUC in legend label,
diagonal chance line for ROC (or baseline rate for PR).

```python
fig, ax = plt.subplots(figsize=(6.5, 6))

for name, d in methods.items():
    fpr, tpr = d["fpr"], d["tpr"]
    ax.plot(fpr, tpr, color=d["color"], linewidth=2.0,
            label=f"{name} (AUC = {d['auc']:.2f})")

# Diagonal chance line
ax.plot([0, 1], [0, 1], linestyle="--", color=NEUTRAL["label"],
        linewidth=1.0, label="Chance")

ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.set_aspect("equal")           # ROC must be square
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.legend(loc="lower right")

figure_title(fig, "Binding-affinity classifier ROC", y=1.00)
save_figure(fig, "fig_roc")
```

Notes:
- ROC: x = FPR, y = TPR, diagonal = chance. **Always square.**
- PR: x = recall, y = precision. Baseline = positive class rate (not 0.5).
- For PR curves, replace the diagonal with `ax.axhline(positive_rate)`.
- Compute FPR/TPR with `sklearn.metrics.roc_curve(y_true, y_score)`.
  Compute AUC with `sklearn.metrics.roc_auc_score`.
- For a single curve, you can omit the legend and put `AUC = 0.94` as
  text annotation in the lower-right via `ax.text(0.95, 0.05, ...)`.

---

## 8. Distribution comparison (histograms + KDE)

Overlaid translucent histograms with KDE curves on top. Useful for
property distributions, score histograms, ablation comparisons.

```python
from scipy import stats

distributions = {
    "ZINC baseline":  {"data": baseline_data,  "color": LINE_PALETTE["blue"]},
    "MolCross gen.":  {"data": molcross_data,  "color": LINE_PALETTE["mustard"]},
    "REINVENT gen.":  {"data": reinvent_data,  "color": LINE_PALETTE["pink"]},
}

fig, ax = plt.subplots(figsize=(10, 5.5))
bin_edges = np.linspace(0, 1, 35)

for name, d in distributions.items():
    c = d["color"]
    # Translucent histogram in the background
    ax.hist(d["data"], bins=bin_edges, density=True, color=c, alpha=0.30,
            edgecolor="none", zorder=2)
    # KDE curve on top
    kde = stats.gaussian_kde(d["data"], bw_method=0.25)
    xs = np.linspace(0, 1, 400)
    ax.plot(xs, kde(xs), color=c, linewidth=2.0, label=name, zorder=3)

ax.set_xlabel("Quantitative estimate of drug-likeness (QED)")
ax.set_ylabel("Density")
ax.set_xlim(0, 1)
ax.set_ylim(bottom=0)
ax.legend(loc="upper left")

figure_title(fig, "QED distribution across generation methods", y=1.00)
save_figure(fig, "fig_distributions")
```

Notes:
- `density=True` normalizes so the histograms are directly comparable
  even when sample sizes differ.
- `bw_method=0.25` is a moderate smoothing bandwidth for KDE; lower
  (0.15) for sharp distributions, higher (0.5) for very noisy data.
- For 2+ distributions, share `bin_edges` so the visual comparison is
  honest. Define once, pass to every `ax.hist()` call.
- For very different sample sizes, consider adding `n = ...` to the
  legend label so reviewers know what they're seeing.
- For 5+ distributions, switch to small multiples (one panel per
  distribution) instead of overlay — overlap becomes illegible.

---

## 9. Box / violin plot with overlaid points

Soft pastel filled boxes, dark median line, individual points jittered
on top. Use when sample sizes per group are small (n ≤ ~30 per group)
and you want to show the underlying spread.

```python
methods = ["AdamW", "Lion", "Muon", "AdaptiveMuon"]
colors_box = [BAR_PALETTE["coral"], BAR_PALETTE["peach"],
              BAR_PALETTE["dark_gray"], LINE_PALETTE["sage"]]
data_per_method = [adamw_losses, lion_losses, muon_losses, adaptmuon_losses]

fig, ax = plt.subplots(figsize=(9, 5.5))
positions = np.arange(len(methods))

bp = ax.boxplot(
    data_per_method, positions=positions, widths=0.55,
    patch_artist=True, showfliers=False,
    medianprops=dict(color=NEUTRAL["title"], linewidth=1.6),
    whiskerprops=dict(color=NEUTRAL["spine"], linewidth=1.0),
    capprops=dict(color=NEUTRAL["spine"], linewidth=1.0),
    boxprops=dict(linewidth=0),  # no box border; fill carries the color
)
for patch, c in zip(bp["boxes"], colors_box):
    patch.set_facecolor(c)
    patch.set_alpha(0.65)

# Overlay individual data points with horizontal jitter
for i, (vals, c) in enumerate(zip(data_per_method, colors_box)):
    jitter = np.random.uniform(-0.12, 0.12, len(vals))
    ax.scatter(np.full_like(vals, i) + jitter, vals, color=c,
               s=18, alpha=0.85, linewidth=0.5,
               edgecolor=NEUTRAL["title"], zorder=4)

ax.set_xticks(positions)
ax.set_xticklabels(methods)
ax.set_ylabel("Final validation loss (8 seeds)")

figure_title(fig, "Optimizer stability on held-out validation", y=1.00)
save_figure(fig, "fig_boxplot")
```

For **violin plots** instead, replace `ax.boxplot(...)` with:

```python
vp = ax.violinplot(data_per_method, positions=positions, widths=0.7,
                    showmeans=False, showmedians=True, showextrema=False)
for body, c in zip(vp["bodies"], colors_box):
    body.set_facecolor(c)
    body.set_alpha(0.55)
    body.set_edgecolor("none")
vp["cmedians"].set_color(NEUTRAL["title"])
vp["cmedians"].set_linewidth(1.6)
```

Notes:
- Hide outliers with `showfliers=False` because the overlaid scatter
  shows them anyway — having both is visually redundant.
- For large n per group (>30), drop the scatter overlay; just the box.
- `boxprops=dict(linewidth=0)` produces a flat fill with no border.
  Cleaner than colored borders.

---

## 10. Scaling-law plot (per-run curves + compute-efficient frontier)

The iconic Kaplan/Chinchilla figure: one full loss-vs-compute training
curve per model size — each run descends steeply, kisses the frontier
near its compute-optimal point, then peels away into a diminishing-returns
plateau. Runs are colored by parameter count on a log colorbar
(`CMAP_GRADIENT`), and the dashed compute-efficient frontier is fit to
the lower envelope of the family.

```python
import matplotlib as mpl

# One training run per model size: arrays of (compute, loss).
# Synthetic Kaplan-form placeholder, L(N, D) = (Nc/N)^aN + (Dc/D)^aD with
# C = 6*N*D — with REAL data, replace this loop with each run's logged
# (compute, loss) trace and keep everything below unchanged.
Nc, aN, Dc, aD = 8.8e13, 0.076, 5.0e9, 0.30
Ns = np.logspace(5, 11, 8)
runs = []  # (N, C, L) per run
for N in Ns:
    plateau = (Nc / N) ** aN
    D_star = Dc * ((aN / aD) * plateau) ** (-1 / aD)  # frontier tangency
    D = D_star * np.logspace(-2.5, 1.8, 300)
    C = 6.0 * N * D
    L = plateau + (Dc / D) ** aD
    runs.append((N, C, L * np.exp(np.random.normal(0, 0.004, len(L)))))

norm = mpl.colors.LogNorm(Ns.min(), Ns.max())
fig, ax = plt.subplots(figsize=(9, 5.8))
for N, C, L in runs:
    ax.plot(C, L, color=CMAP_GRADIENT(norm(N)), linewidth=1.2, alpha=0.95,
            zorder=2)

# Frontier = lower envelope of the family, fit as a line in log-log space.
# Fit ONLY where an interior run is optimal: at the extremes the envelope
# is biased high because no smaller/larger model exists there.
C_dense = np.logspace(np.log10(min(C.min() for _, C, _ in runs)),
                      np.log10(max(C.max() for _, C, _ in runs)), 600)
losses = np.array([np.interp(C_dense, C, L, left=np.inf, right=np.inf)
                   for _, C, L in runs])
env, which = losses.min(axis=0), losses.argmin(axis=0)
interior = (which > 0) & (which < len(runs) - 1)
slope, icpt = np.polyfit(np.log10(C_dense[interior]),
                         np.log10(env[interior]), 1)
ax.plot(C_dense, 10**icpt * C_dense**slope, color=NEUTRAL["ink"],
        linestyle="--", linewidth=1.4, zorder=3,
        label=fr"Frontier: $L \propto C^{{{slope:.3f}}}$")

ax.set_xscale("log")
ax.set_yscale("log")
# Loss spans <1 decade: plain numbers beat 2×10⁰-style log tick labels
plain_log_ticks(ax, [2, 3, 4, 6, 9])
ax.set_xlabel("Training compute (FLOPs)")
ax.set_ylabel("Validation loss")
ax.legend(loc="upper right")

soft_colorbar(fig, ax, norm=norm, cmap=CMAP_GRADIENT,
              label="Model size (parameters)", pad=0.02)

figure_title(fig, "Loss curves trace the compute-efficient frontier", y=1.00)
save_figure(fig, "fig_scaling")
```

Notes:
- The synthetic `aD=0.30` is deliberately steeper than Kaplan's literal
  ~0.095 — with the real exponents the per-run deviation from the
  frontier is a few percent and the fan is invisibly tight. Real logged
  runs have real curvature; this only matters for placeholder data.
- ~7–9 runs is the sweet spot. More than ~10 and descending curves
  tangle with neighbors' plateaus; fewer than ~6 reads as disconnected.
- The interior-argmin mask on the envelope fit matters: the leftmost and
  rightmost compute decades are "owned" by the smallest/largest run with
  no competitor, so including them drags the fitted line off the true
  frontier (it ends up cutting through the bundle).
- A colorbar replaces the legend for the runs — never put 8 entries in a
  legend box. Keep the legend for the frontier line only.
- When the y-range spans less than a decade, matplotlib's `2×10⁰`-style
  log labels waste ink: pick round numbers with `plain_log_ticks()`,
  as above.
- Always log-log for power laws — a straight line is the reader's most
  important visual cue.

**Variant: final-loss scatter + per-family power-law fits.** When you
only have one terminal loss per run (no training traces), fall back to
scatter + fitted lines, one color per family:

```python
ax.scatter(compute, loss_obs, color=c, s=45, zorder=3,
           edgecolor="white", linewidth=0.8)
cf = np.logspace(np.log10(compute.min() * 0.7),
                 np.log10(compute.max() * 1.5), 100)
ax.plot(cf, C0 * cf ** (-alpha), color=c, linewidth=1.6, alpha=0.85,
        label=fr"Dense: $L \propto C^{{-{alpha:.3f}}}$")
```

Fit the exponent with `np.polyfit` on `log(loss)` vs `log(compute)`
(linear in log-log space), one fit per family.

---

## 11. Parity / calibration plot

Predicted vs experimental values, square axis, y=x diagonal, fit stats
in a monospace box. Standard for QSAR, regression diagnostics, and any
"do my predictions match reality" plot.

```python
residuals = y_pred - y_true
ss_res = np.sum(residuals ** 2)
ss_tot = np.sum((y_true - y_true.mean()) ** 2)
r2 = 1 - ss_res / ss_tot
mae = np.mean(np.abs(residuals))
rmse = np.sqrt(np.mean(residuals ** 2))

fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_true, y_pred, color=LINE_PALETTE["blue"], s=22,
           alpha=0.55, linewidth=0, zorder=2)

# y = x diagonal
lim_lo, lim_hi = -3.0, 4.0
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], linestyle="--",
        color=NEUTRAL["label"], linewidth=1.2, label="y = x", zorder=3)

ax.set_xlim(lim_lo, lim_hi)
ax.set_ylim(lim_lo, lim_hi)
ax.set_aspect("equal")
ax.set_xlabel("Experimental logP")
ax.set_ylabel("Predicted logP")

# Stats box, upper-left, monospace for column alignment
stats_text = (f"$R^2$ = {r2:.3f}\n"
              f"MAE = {mae:.3f}\n"
              f"RMSE = {rmse:.3f}\n"
              f"n = {len(y_true)}")
ax.text(0.04, 0.96, stats_text, transform=ax.transAxes,
        fontsize=9.5, color=NEUTRAL["title"], va="top", ha="left",
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white",
                  edgecolor=NEUTRAL["spine"], linewidth=0.7))

ax.legend(loc="lower right")
figure_title(fig, "logP prediction: parity with experimental values", y=1.00)
save_figure(fig, "fig_parity")
```

Notes:
- Parity plots **must** be square (`ax.set_aspect("equal")`) and
  identical limits on both axes. Reviewers will flag non-square parity
  plots as misleading.
- For colored scatter by an additional variable (e.g., molecule size),
  pass `c=size, cmap=CMAP_SEQUENTIAL` and add a colorbar.
- The `family="monospace"` for the stats box keeps the values
  column-aligned. The actual numbers will be in Liberation Mono or
  DejaVu Sans Mono.
- Optional ±error band: `ax.fill_between([lim_lo, lim_hi],
  [lim_lo - rmse, lim_hi - rmse], [lim_lo + rmse, lim_hi + rmse],
  color=NEUTRAL["band"], alpha=0.3, zorder=1)`.

---

## 12. 2D embedding scatter (t-SNE / UMAP / PCA)

Categorical clusters in 2D space, no informative tick labels (embedding
coordinates are arbitrary), legend with a title.

```python
clusters = {
    "Aromatic":    {"center": (-3, 2),  "n": 180, "color": LINE_PALETTE["blue"]},
    "Aliphatic":   {"center": (2, -3),  "n": 160, "color": LINE_PALETTE["mustard"]},
    "Heterocycle": {"center": (-2, -3), "n": 140, "color": LINE_PALETTE["sage"]},
    "Carbonyl":    {"center": (3, 3),   "n": 150, "color": LINE_PALETTE["pink"]},
    "Halide":      {"center": (0, 0),   "n": 100, "color": MULTILINE_PALETTE["dark_green"]},
}

fig, ax = plt.subplots(figsize=(7.5, 6.5))
for name, d in clusters.items():
    cx, cy = d["center"]
    xs = cx + np.random.normal(0, 1.0, d["n"])
    ys = cy + np.random.normal(0, 1.0, d["n"])
    ax.scatter(xs, ys, color=d["color"], s=22, alpha=0.55,
               linewidth=0, label=name, zorder=2)

ax.set_xlabel("UMAP dim 1")
ax.set_ylabel("UMAP dim 2")
ax.set_aspect("equal")
ax.tick_params(labelleft=False, labelbottom=False)  # hide coord values
ax.legend(loc="upper left", title="Functional group", markerscale=1.5)

figure_title(fig, "Molecular embedding: ChemBERTa hidden states", y=1.00)
save_figure(fig, "fig_embedding")
```

Notes:
- Hide tick labels — UMAP/t-SNE coordinates are arbitrary and reading
  them off the axes is meaningless. Keep only the spines so readers
  know it's a 2D space.
- `aspect="equal"` is essential; aspect distortion makes clusters look
  artificially separated or merged.
- For >6 categories switch to small multiples (one cluster per panel)
  or use shape variation (`marker="o", "s", "^", "D"`) layered on color.
- For continuous coloring (e.g., colored by molecular weight): pass
  `c=values, cmap=CMAP_SEQUENTIAL`, omit `label=`, and add a colorbar.
- Add `markerscale=1.5` to the legend so legend markers are visible
  even when actual scatter markers are small.

---

## 13. ECDF / CDF plot

Cumulative distribution comparison, no binning artifacts, sensitive
to tail behavior. Use when distributions are heavy-tailed or when the
question is about a quantile (median, 90th percentile, etc.).

```python
methods = {
    "AdamW":         {"data": adamw_losses,        "color": BAR_PALETTE["coral"]},
    "Lion":          {"data": lion_losses,         "color": BAR_PALETTE["peach"]},
    "Muon":          {"data": muon_losses,         "color": BAR_PALETTE["dark_gray"]},
    "AdaptiveMuon":  {"data": adaptive_muon_losses,"color": LINE_PALETTE["sage"]},
}

fig, ax = plt.subplots(figsize=(9, 5.5))
for name, d in methods.items():
    sorted_d = np.sort(d["data"])
    ecdf_y = np.arange(1, len(sorted_d) + 1) / len(sorted_d)
    ax.plot(sorted_d, ecdf_y, color=d["color"], linewidth=2.0,
            label=name, drawstyle="steps-post")

ax.set_xlabel("Per-token loss")
ax.set_ylabel("Cumulative fraction of tokens")
ax.set_xlim(left=0)
ax.set_ylim(0, 1.02)
ax.legend(loc="lower right")

figure_title(fig, "Per-token loss ECDF on validation set", y=1.00)
save_figure(fig, "fig_ecdf")
```

Notes:
- `drawstyle="steps-post"` produces the proper step-function ECDF.
  Remove this for a smoothed visual approximation.
- For tail comparison, use a log x-axis: `ax.set_xscale("log")`.
- Annotate quantiles with horizontal lines:
  `ax.axhline(0.5, color=NEUTRAL["spine"], linewidth=0.6, linestyle=":")`.
- ECDFs are strictly preferable to histograms when comparing 3+
  distributions — no bin-edge dependency, no overplotting issues.
- For paired data, plot the *difference* between two ECDFs to highlight
  where they diverge.

---

## 14. Forest plot / dot-and-whisker

Effect sizes with confidence intervals across many interventions. The
go-to plot for ablation summaries, meta-analyses, and safety eval
overviews. The y-axis is categorical (intervention name), the x-axis
is the effect size with CI as horizontal error bars.

```python
interventions = [
    # (name, effect, std_error)
    ("Baseline (Sonnet 4)",           0.000, 0.000),
    ("Constitutional SDF",           -0.382, 0.042),
    ("Constitutional SDF + stories", -0.469, 0.038),
    ("Difficult advice (3M)",        -0.046, 0.058),
    ("PM filtering (23M)",           -0.048, 0.045),
    ("Refusal pretraining",          -0.215, 0.052),
    ("Mixed SDF + RLHF",             -0.420, 0.035),
]

names = [r[0] for r in interventions]
effects = np.array([r[1] for r in interventions])
errors = np.array([r[2] for r in interventions])
y_pos = np.arange(len(names))

fig, ax = plt.subplots(figsize=(10, 6.5))
ax.axvline(0, linestyle="--", color=NEUTRAL["label"], linewidth=1.0, zorder=1)

for y, eff, err in zip(y_pos, effects, errors):
    # Color by significance and direction
    color = (BAR_PALETTE["dark_gray"] if eff < -0.1
             else BAR_PALETTE["coral"] if eff > 0.05
             else NEUTRAL["tick"])  # ambiguous / null effect
    ax.errorbar(eff, y, xerr=err * 1.96, fmt="o", color=color,
                markersize=8, capsize=4, elinewidth=1.5, capthick=1.2,
                zorder=3)

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.invert_yaxis()  # first row at top
ax.set_xlabel("Effect on misalignment score (Δ vs. baseline, 95% CI)")

figure_title(fig, "Intervention effects on agentic misalignment", y=1.00)
save_figure(fig, "fig_forest")
```

Notes:
- `xerr = std_error * 1.96` produces ~95% CIs from standard errors.
  Use the appropriate multiplier for your statistic — for t-distribution
  with small n, use scipy.stats.t.ppf(0.975, df).
- Color encoding lets reviewers scan for "which interventions work":
  dark = significant beneficial, coral = significant harmful, gray =
  ambiguous (CI crosses zero).
- For many interventions (>12), split into facet panels by category
  (training data, post-training method, model size, etc.).
- The dashed zero line is required — without it the reader can't
  judge significance.
- Group interventions visually with horizontal separators if the list
  has clear sub-categories.

---

## 15. Horizontal bar chart (sorted ranking)

For comparing many methods (8+) where vertical bars would have
overlapping or rotated x-labels. Sort by the metric, highlight the
top performer with the accent color, mute the rest. Bars get the same
rounded look as vertical ones — square left edge flush against x=0,
softly rounded right end — via `rounded_hbars()`.

```python
methods = [
    ("Random", 0.32), ("MACCS keys", 0.51), ("Morgan r=2", 0.68),
    ("Morgan r=3", 0.72), ("ECFP4", 0.71), ("ECFP6", 0.74),
    ("Atom-pair", 0.69), ("RDKit topological", 0.65), ("Avalon", 0.67),
    ("Mordred", 0.78), ("ChemBERTa", 0.82), ("MolBERT", 0.81),
    ("SPAN/AFP", 0.86), ("MolCross", 0.89),
]
methods.sort(key=lambda x: x[1])  # ascending: best ends at the top

names = [m[0] for m in methods]
scores = [m[1] for m in methods]
n = len(methods)

# Top performer: coral. Next two: peach. Rest: light gray.
colors = [BAR_PALETTE["coral"] if i == n - 1
          else BAR_PALETTE["peach"] if i >= n - 3
          else BAR_PALETTE["light"]
          for i in range(n)]

fig, ax = plt.subplots(figsize=(9, 7))
y_pos = np.arange(n)

# Set limits BEFORE rounded_hbars for accurate corner geometry
ax.set_xlim(0, 1.0)
ax.set_ylim(-0.7, n - 0.3)

# Each bar gets its own color → one rounded_hbars call per bar
for y, s, c in zip(y_pos, scores, colors):
    rounded_hbars(ax, [y], [s], height=0.65, color=c, radius_frac=0.20)

# Value labels at the end of each bar
for y, s in zip(y_pos, scores):
    ax.text(s + 0.012, y, f"{s:.3f}", va="center", ha="left",
            fontsize=9.5, color=NEUTRAL["title"])

ax.set_yticks(y_pos)
ax.set_yticklabels(names)
ax.set_xlabel("Validation AUC")

figure_title(fig, "Binding-affinity classifier by molecular representation",
             y=1.00)
save_figure(fig, "fig_hbar")
```

Notes:
- Always sort by the metric — unsorted horizontal bars are unreadable.
- The "highlight top, mute rest" coloring tells the reader where the
  story is. For comparison plots without a single winner, use the
  full palette evenly across bars.
- Value labels at the end of each bar save the reader from estimating
  positions. Use `f"{s:.3f}"` for 3 decimals, `f"{s:.1%}"` for percent.
- `rounded_hbars()` is the horizontal counterpart of `rounded_bars()`:
  square LEFT edge (flush against x=0 baseline), rounded RIGHT corners.
  The corner radius is `radius_frac * height`, and is converted to
  display pixels so corners look circular regardless of axis aspect.
- For per-bar colors, call `rounded_hbars` once per bar (one-element
  arrays for `y` and `widths`). For a single color across all bars,
  pass the full arrays in one call.
- For 20+ methods, consider faceting by category (e.g., classical
  fingerprints vs. neural fingerprints) or showing only top-N.

---

## 16. Pareto front / trade-off plot

Scatter of competing methods on two axes (e.g., accuracy vs. latency,
quality vs. cost), with the Pareto frontier traced. The story is "for
this latency, what's the best accuracy you can get" — the frontier
shows the achievable boundary.

```python
# (name, params_M, accuracy, latency_ms, family)
models = [
    ("DistilBERT",       66,  0.74, 12, "dense"),
    ("BERT-base",        110, 0.79, 22, "dense"),
    ("DeBERTa-v3-base",  184, 0.84, 28, "dense"),
    ("MoE-S",            80,  0.81, 14, "moe"),
    ("MoE-M",            200, 0.85, 24, "moe"),
    ("MoE-L",            500, 0.89, 42, "moe"),
    ("ours-base",        120, 0.86, 18, "ours"),
    ("ours-large",       310, 0.91, 35, "ours"),
]

family_colors = {
    "dense": LINE_PALETTE["blue"],
    "moe":   LINE_PALETTE["mustard"],
    "ours":  BAR_PALETTE["coral"],
}
family_labels = {"dense": "Dense baselines", "moe": "MoE",
                 "ours": "Ours (this work)"}

fig, ax = plt.subplots(figsize=(9, 6))
for fam, color in family_colors.items():
    fam_models = [m for m in models if m[4] == fam]
    lat = [m[3] for m in fam_models]
    acc = [m[2] for m in fam_models]
    sizes = [40 + m[1] * 1.5 for m in fam_models]  # encode params in size
    ax.scatter(lat, acc, s=sizes, color=color, alpha=0.75,
               edgecolor="white", linewidth=1.0, zorder=3)

# Pareto frontier: sort by x, walk through keeping max y so far
all_sorted = sorted(models, key=lambda m: m[3])
frontier, best = [], -np.inf
for m in all_sorted:
    if m[2] > best:
        frontier.append(m)
        best = m[2]
ax.plot([m[3] for m in frontier], [m[2] for m in frontier],
        color=NEUTRAL["label"], linestyle=":", linewidth=1.5,
        alpha=0.7, zorder=2)

# Annotate selected points
for m in models:
    if m[0] in {"ours-large", "MoE-L"}:
        ax.annotate(m[0], (m[3], m[2]),
                    xytext=(8, 6), textcoords="offset points",
                    fontsize=9, color=NEUTRAL["title"])

ax.set_xlabel("Inference latency (ms / sample)")
ax.set_ylabel("Validation accuracy")

# IMPORTANT: build legend handles manually with FIXED marker size.
# matplotlib's auto-legend pulls scatter sizes from `s=...` which
# encodes parameters here — that would make legend dots huge and
# overlapping. Custom Line2D handles decouple "color encoding" in the
# legend from "size encoding" in the plot.
from matplotlib.lines import Line2D
handles = [
    Line2D([0], [0], marker="o", color="w",
           markerfacecolor=family_colors[fam],
           markersize=9, markeredgecolor="white",
           markeredgewidth=1.0, label=family_labels[fam])
    for fam in family_colors
]
handles.append(Line2D([0], [0], color=NEUTRAL["label"],
                      linestyle=":", linewidth=1.5, label="Pareto frontier"))
ax.legend(handles=handles, loc="lower right", labelspacing=0.6)

figure_title(fig, "Accuracy vs. latency trade-off across model families",
             y=1.00)
save_figure(fig, "fig_pareto")
```

Notes:
- Marker size encoding parameters lets you fit 3 dimensions into 2D.
  Document the encoding either in the caption or with a small size
  legend (matplotlib doesn't draw size legends by default).
- **Whenever you use `s=` in scatter for a data encoding, build the
  legend by hand with `Line2D` handles at a fixed `markersize`.**
  Otherwise matplotlib pulls sizes from `s=` and the legend dots
  inherit those — huge, overlapping, unreadable. This applies to any
  size-encoded scatter, not just Pareto plots.
- The Pareto frontier here is the *upper-left* envelope (high accuracy,
  low latency). Adjust the comparison direction if "lower is better"
  on both axes.
- Anchor your method's win visually: place it at the top-left of the
  frontier (best accuracy at lowest latency for that level).
- Annotate only the 2–3 most important points; annotating everything
  is illegible.
- For >3 families, switch to consistent markers + colors and place the
  family in the legend; don't add more axes.
