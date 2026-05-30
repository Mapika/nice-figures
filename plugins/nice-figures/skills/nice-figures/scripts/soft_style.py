"""soft_style.py — matplotlib helpers for the nice-figures skill.

Recreates the visual register used by AI alignment / research-blog figures:
  - White background (conference-ready) or warm cream (blog-style)
  - Bold display title above the panel(s)
  - Soft pastel palettes for line plots; warm-earth palette for bar charts
  - Minimal axes (bottom + left spines only), no grid
  - Smoothed trend lines with shaded confidence bands
  - "↓better" / "↑better" badges in the corner

Public API:
    configure_style(cream_bg=False)     — apply rcParams; defaults to white bg
    LINE_PALETTE, BAR_PALETTE, MULTILINE_PALETTE — dicts of named hex colors
    figure_title(fig, text)             — bold display title, sits above panels
    panel_subtitle(ax, text)            — light-gray subtitle above a single panel
    better_badge(ax, direction)         — ↓better / ↑better corner badge
    smooth_curve(x, y, frac)            — Gaussian-kernel smoothed curve
    rolling_band(x, y, frac, k)         — smoothed mean ± k·σ for shaded bands
    save_figure(fig, name, formats)     — multi-format export, transparent-safe

This module has no dependencies beyond matplotlib + numpy. For real LOWESS
smoothing on noisy data, prefer statsmodels.nonparametric.smoothers_lowess —
the smoother here is intentionally lightweight.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


# ---------------------------------------------------------------- palettes --

# Line plots with confidence bands (matches the "Rate of alignment failures
# over RL steps" archetype: blue / mustard / sage / pink).
LINE_PALETTE: dict[str, str] = {
    "blue":    "#6A9CC9",
    "mustard": "#D9A04E",
    "sage":    "#7AA890",
    "pink":    "#C45D7C",
}

# Grouped bar charts (matches "Automated auditing scores by model" and
# "Agentic misalignment evals": warm coral + peach + dark gray + light olive).
BAR_PALETTE: dict[str, str] = {
    "coral":     "#D8704C",
    "peach":     "#EBC5A8",
    "dark_gray": "#5C5B58",
    "light":     "#C9C5BA",
}

# Multi-line plots with categorical series (matches "Misalignment over
# RL training": teal / pink / dark green / mustard / blue).
MULTILINE_PALETTE: dict[str, str] = {
    "teal":       "#3FAB91",
    "pink":       "#E69BB0",
    "dark_green": "#3B6E4D",
    "mustard":    "#E0B057",
    "blue":       "#6A9CC9",
}

# Single-color accents for baseline lines, annotation arrows, etc.
NEUTRAL: dict[str, str] = {
    "title":     "#1B1A18",
    "label":     "#6B6960",
    "tick":      "#8C8A82",
    "spine":     "#B8B5AC",
    "band":      "#D8D5CC",
    "white":     "#FFFFFF",
    "cream_bg":  "#E8E5D7",   # original beige outer
    "cream_pnl": "#F4F1E6",   # original cream panel
}


# Custom colormaps in the soft register, built lazily so importing this
# module doesn't pay the matplotlib registration cost unless needed.
def _build_cmaps():
    from matplotlib.colors import LinearSegmentedColormap
    sequential = LinearSegmentedColormap.from_list(
        "soft_sequential",
        ["#FFFFFF", "#F4E8DC", "#EBC5A8", "#D8704C", "#8C3520"],
        N=256,
    )
    diverging = LinearSegmentedColormap.from_list(
        "soft_diverging",
        ["#3A6E8C", "#6A9CC9", "#C4D5DD",
         "#F4E8DC",
         "#EBC5A8", "#D8704C", "#8C3520"],
        N=256,
    )
    return sequential, diverging


CMAP_SEQUENTIAL, CMAP_DIVERGING = _build_cmaps()


# ---------------------------------------------------------------- styling --

def configure_style(cream_bg: bool = False) -> None:
    """Apply the soft-research style globally via rcParams.

    Args:
        cream_bg: If True, use the warm cream/beige background of the
            original Anthropic blog figures. Default False (white background),
            which is what most academic conferences expect.
    """
    if cream_bg:
        outer_bg = NEUTRAL["cream_bg"]
        panel_bg = NEUTRAL["cream_pnl"]
    else:
        outer_bg = "white"
        panel_bg = "white"

    plt.rcParams.update({
        # backgrounds
        "figure.facecolor":   outer_bg,
        "axes.facecolor":     panel_bg,
        "savefig.facecolor":  outer_bg,
        "savefig.edgecolor":  "none",

        # fonts — Inter/Helvetica/Arial preferred, DejaVu as fallback
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Inter", "Helvetica Neue", "Helvetica",
                                "Arial", "Liberation Sans", "DejaVu Sans"],
        "mathtext.fontset":    "dejavusans",  # math in sans, not serif default
        "font.size":           11,
        "axes.titlesize":      13,
        "axes.titleweight":    "normal",
        "axes.titlecolor":     NEUTRAL["label"],
        "axes.titlepad":       12,
        "axes.labelsize":      11,
        "axes.labelcolor":     NEUTRAL["label"],
        "axes.labelweight":    "normal",
        "axes.labelpad":       8,
        "xtick.labelsize":     9.5,
        "ytick.labelsize":     9.5,
        "xtick.color":         NEUTRAL["tick"],
        "ytick.color":         NEUTRAL["tick"],
        "legend.fontsize":     9.5,
        "figure.titlesize":    18,
        "figure.titleweight":  "bold",

        # spines: only bottom + left in soft gray
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.spines.bottom":  True,
        "axes.spines.left":    True,
        "axes.edgecolor":      NEUTRAL["spine"],
        "axes.linewidth":      0.9,

        # ticks: short, outward, light
        "xtick.major.width":   0.8,
        "ytick.major.width":   0.8,
        "xtick.major.size":    3.0,
        "ytick.major.size":    3.0,
        "xtick.minor.width":   0.5,
        "ytick.minor.width":   0.5,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "xtick.major.pad":     5,
        "ytick.major.pad":     5,

        # no grid
        "axes.grid":           False,

        # lines & markers
        "lines.linewidth":     2.0,
        "lines.markersize":    4.0,
        "lines.solid_capstyle": "round",

        # legend: thin gray rounded box, no shadow
        "legend.frameon":      True,
        "legend.facecolor":    panel_bg,
        "legend.edgecolor":    NEUTRAL["spine"],
        "legend.framealpha":   1.0,
        "legend.fancybox":     True,
        "legend.borderpad":    0.7,
        "legend.handletextpad": 0.6,
        "legend.labelcolor":   NEUTRAL["title"],
    })


# -------------------------------------------------------- titles & badges --

def figure_title(fig, text: str, y: float | None = None) -> None:
    """Bold display title above the figure, Anthropic-style.

    Args:
        fig: The Figure object.
        text: Title text (typically a short sentence-case phrase).
        y: Vertical position in figure coordinates. Default None lets
            matplotlib choose, but you may want 0.97–1.02 depending on
            whether you've reserved space at the top.
    """
    kwargs = dict(fontsize=18, fontweight="bold",
                  color=NEUTRAL["title"], ha="center")
    if y is not None:
        kwargs["y"] = y
    fig.suptitle(text, **kwargs)


def panel_subtitle(ax, text: str) -> None:
    """Light-gray subtitle above a single panel (e.g. 'Blackmail')."""
    ax.set_title(text, color=NEUTRAL["label"], fontsize=12,
                 fontweight="normal", pad=10)


def better_badge(ax, direction: str = "down",
                 loc: str = "upper right") -> None:
    """The '↓better' / '↑better' rounded badge in a corner of an axis.

    Args:
        ax: Target axis.
        direction: 'down' (lower is better) or 'up' (higher is better).
        loc: 'upper right', 'upper left', 'lower right', or 'lower left'.
    """
    arrow = {"down": "↓", "up": "↑"}[direction]
    text = f"{arrow}better"
    positions = {
        "upper right": (0.97, 0.95, "right", "top"),
        "upper left":  (0.03, 0.95, "left",  "top"),
        "lower right": (0.97, 0.05, "right", "bottom"),
        "lower left":  (0.03, 0.05, "left",  "bottom"),
    }
    x, y, ha, va = positions[loc]
    ax.text(
        x, y, text, transform=ax.transAxes,
        ha=ha, va=va, fontsize=9, color=NEUTRAL["label"],
        bbox=dict(
            boxstyle="round,pad=0.45,rounding_size=0.4",
            facecolor=NEUTRAL["white"],
            edgecolor=NEUTRAL["spine"],
            linewidth=0.7,
        ),
    )


# -------------------------------------------------------- smoothing utils --

def smooth_curve(x, y, frac: float = 0.25, n_out: int = 200):
    """Gaussian-kernel smoothed curve through (x, y).

    Lightweight LOWESS alternative — no scipy dependency. For heavy-tailed
    or non-uniform x, prefer statsmodels.nonparametric.smoothers_lowess.

    Args:
        x, y: Input arrays of same length.
        frac: Bandwidth as fraction of n. Higher = smoother.
        n_out: Number of output points on a uniform grid.

    Returns:
        (x_smooth, y_smooth) numpy arrays.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return x, y

    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    window = max(int(frac * n), 3)
    sigma = max(window / 2.5, 1e-6)

    x_smooth = np.linspace(x_s.min(), x_s.max(), n_out)
    y_smooth = np.empty_like(x_smooth)
    for i, xi in enumerate(x_smooth):
        w = np.exp(-((x_s - xi) ** 2) / (2 * sigma ** 2))
        y_smooth[i] = np.sum(w * y_s) / np.sum(w)
    return x_smooth, y_smooth


def rolling_band(x, y, frac: float = 0.25, k: float = 1.0, n_out: int = 200):
    """Smoothed mean ± k·σ for shaded confidence bands.

    Args:
        x, y: Input arrays of same length.
        frac: Bandwidth as fraction of n.
        k: Half-width in standard-deviation units. k=1 ≈ 68% band,
            k=1.96 ≈ 95% band.
        n_out: Number of output points.

    Returns:
        (x_smooth, lower, upper, mean).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 3:
        return x, y, y, y

    order = np.argsort(x)
    x_s, y_s = x[order], y[order]

    window = max(int(frac * n), 3)
    sigma = max(window / 2.5, 1e-6)

    x_smooth = np.linspace(x_s.min(), x_s.max(), n_out)
    y_mean = np.empty_like(x_smooth)
    y_std = np.empty_like(x_smooth)
    for i, xi in enumerate(x_smooth):
        w = np.exp(-((x_s - xi) ** 2) / (2 * sigma ** 2))
        w_norm = w / np.sum(w)
        y_mean[i] = np.sum(w_norm * y_s)
        y_std[i] = np.sqrt(np.sum(w_norm * (y_s - y_mean[i]) ** 2))
    return x_smooth, y_mean - k * y_std, y_mean + k * y_std, y_mean


# ------------------------------------------------------------- exporting --

def save_figure(fig, name: str, formats=("pdf", "png"), dpi: int = 300):
    """Save figure to multiple formats. Preserves whatever facecolor the
    figure was built with (white or cream)."""
    bg = fig.get_facecolor()
    for ext in formats:
        out = f"{name}.{ext}"
        fig.savefig(out, dpi=dpi, bbox_inches="tight",
                    facecolor=bg, edgecolor="none")
        print(f"Saved {out}")


# ----------------------------------------------------------- rounded bars --

def _rounded_top_path(ax, left: float, right: float, top: float,
                      baseline: float, r_px: float):
    """Path for one bar: square bottom at ``baseline``, rounded top at ``top``.

    ``r_px`` is the target corner radius in display pixels (capped to half the
    bar's pixel width/height). Geometry is computed in DISPLAY pixels through the
    axes transform, then mapped back to data coordinates. Because nothing is
    anchored at ``y=0`` and no point is transformed at the origin, this is valid
    on **log axes** as well as linear, and the corners stay circular regardless
    of scale, aspect, or which subplot the axis is. Returns ``None`` for a
    degenerate (zero-extent) bar so the caller can skip it.
    """
    from matplotlib.path import Path

    trans = ax.transData
    bl = trans.transform((left,  baseline))
    br = trans.transform((right, baseline))
    tl = trans.transform((left,  top))
    tr = trans.transform((right, top))
    w_pix = abs(br[0] - bl[0])
    h_pix = abs(tl[1] - bl[1])
    if w_pix <= 0 or h_pix <= 0:
        return None

    r = min(r_px, w_pix / 2.0, h_pix / 2.0)
    sgn = 1.0 if tl[1] >= bl[1] else -1.0          # bar grows up (or down)
    verts_px = [
        (bl[0],     bl[1]),                # MOVETO  bottom-left
        (tl[0],     tl[1] - sgn * r),      # LINETO  up the left side
        (tl[0],     tl[1]),                # CURVE3  control (top-left corner)
        (tl[0] + r, tl[1]),                # CURVE3  endpoint
        (tr[0] - r, tr[1]),                # LINETO  across the top
        (tr[0],     tr[1]),                # CURVE3  control (top-right corner)
        (tr[0],     tr[1] - sgn * r),      # CURVE3  endpoint
        (br[0],     br[1]),                # LINETO  down the right side
        (0.0,       0.0),                  # CLOSEPOLY (vertex ignored)
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    verts_data = trans.inverted().transform(verts_px)
    return Path(verts_data, codes)


def rounded_bars(ax, x, heights, width: float, color: str,
                 radius_frac: float = 0.18, radius_pt: float | None = None,
                 baseline=None, label=None, zorder: float = 2, **kwargs):
    """Draw bars with square bottoms and rounded top corners.

    The visual signature of the Anthropic alignment-figure bar charts — top
    corners softly rounded, bottom corners square on the baseline. A drop-in
    replacement for ``ax.bar()`` that works on **linear or log** y-axes and in
    any subplot, because the rounded top is constructed in display pixels
    through the axes transform rather than from a hard-coded ``y=0`` baseline.

    **Set ax.set_yscale / set_ylim / set_xlim BEFORE calling this** — the corner
    geometry is read from the axes transform at call time. (Calling it after a
    later relimit can stretch the corners.)

    For the legend, use ``matplotlib.patches.Patch(facecolor=..., label=...)``;
    auto-legend handles from PathPatch don't look right.

    Args:
        ax: Target axis.
        x: Bar center x-positions.
        heights: Bar tops (absolute y-values, not deltas).
        width: Bar width in x-data coordinates.
        color: Fill color.
        radius_frac: Corner radius as a fraction of the bar's pixel width
            (0.15–0.22 matches the house look; 0.5 gives a pill cap). Used only
            when ``radius_pt`` is None.
        radius_pt: Corner radius as an **absolute** size in points, overriding
            ``radius_frac``. Prefer this when a figure mixes wide and narrow
            bars (e.g. a single panel vs. grouped subplots): a fixed point
            radius gives every bar in the figure set the *same* visual corner,
            whereas ``radius_frac`` makes narrow bars look less rounded. Capped
            to half the bar's pixel width/height.
        baseline: Bottom of the bars. ``None`` → the axis's current lower
            limit (``ax.get_ylim()[0]``), which is ``0`` on a linear-from-zero
            axis and a safe positive value on a log axis. Pass a scalar to
            anchor all bars, or an array (one per bar) to stack rounded tops
            on top of existing segments.
        label: Optional legend label (attached to the first patch only).
        zorder: Drawing order. Default 2 (behind error bars at zorder 3).
        **kwargs: Forwarded to PathPatch (e.g. alpha=0.8).

    Returns:
        List of PathPatch objects added to the axis.
    """
    from matplotlib.patches import PathPatch

    xs = np.atleast_1d(x).astype(float)
    hs = np.atleast_1d(heights).astype(float)
    if len(xs) != len(hs):
        raise ValueError("x and heights must have the same length")
    if baseline is None:
        baseline = ax.get_ylim()[0]
    bls = np.broadcast_to(np.asarray(baseline, dtype=float), hs.shape)

    if radius_pt is not None:
        r_px = radius_pt * ax.figure.dpi / 72.0          # absolute, width-independent
    else:
        # proportional: fraction of the (shared) bar pixel width
        trans = ax.transData
        x0 = float(xs[0])
        wpx = abs(trans.transform((x0 + width / 2.0, bls[0]))[0]
                  - trans.transform((x0 - width / 2.0, bls[0]))[0])
        r_px = radius_frac * wpx

    patches = []
    for i, (xi, h, b) in enumerate(zip(xs, hs, bls)):
        if h == b:
            continue
        path = _rounded_top_path(ax, xi - width / 2.0, xi + width / 2.0,
                                 float(h), float(b), r_px)
        if path is None:
            continue
        lbl = label if (i == 0 and label is not None) else None
        patch = PathPatch(path, facecolor=color, edgecolor="none",
                          label=lbl, zorder=zorder, **kwargs)
        ax.add_patch(patch)
        patches.append(patch)

    return patches


def _rounded_hbar_path(y_center: float, width: float, height: float,
                       r_x: float, r_y: float):
    """Build a Path for a horizontal bar: square LEFT edge (flush at x=0),
    rounded RIGHT corners (at x=width). Mirror of _rounded_bar_path.

    Returns None for zero/negative width (caller should skip).
    """
    from matplotlib.path import Path

    if width <= 0:
        return None

    bottom = y_center - height / 2
    top = y_center + height / 2

    r_x = min(r_x, width / 2)
    r_y = min(r_y, height / 2)

    verts = [
        (0,           bottom),         # MOVETO  bottom-left (square)
        (width - r_x, bottom),         # LINETO  along the bottom
        (width,       bottom),         # CURVE3  control (bottom-right)
        (width,       bottom + r_y),   # CURVE3  endpoint
        (width,       top - r_y),      # LINETO  up the right side
        (width,       top),            # CURVE3  control (top-right)
        (width - r_x, top),            # CURVE3  endpoint
        (0,           top),            # LINETO  along the top
        (0,           0),              # CLOSEPOLY (vertex ignored)
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CURVE3, Path.CURVE3,
        Path.LINETO,
        Path.CLOSEPOLY,
    ]
    return Path(verts, codes)


def rounded_hbars(ax, y, widths, height: float, color: str,
                  radius_frac: float = 0.20, label=None,
                  zorder: float = 2, **kwargs):
    """Draw horizontal bars with square left edge and rounded right corners.

    Mirror of ``rounded_bars()`` for horizontal orientation. Bars grow
    from x=0 to the right; the left edge sits flush against the baseline
    and the right end has softly rounded corners.

    Like ``rounded_bars()``, corner geometry is computed in display
    pixels so corners look consistently circular regardless of axis
    aspect ratio. **Set ax.set_xlim() and ax.set_ylim() BEFORE calling
    rounded_hbars()** for accurate corners.

    For the legend, use ``matplotlib.patches.Patch(facecolor=..., label=...)``.

    Args:
        ax: Target axis.
        y: Array-like of bar center y-positions.
        widths: Array-like of bar widths (values; assumes baseline at x=0).
        height: Bar height in data y-units.
        color: Fill color.
        radius_frac: Corner radius as a fraction of bar height. 0.18–0.25
            matches the Anthropic look.
        label: Optional legend label (attached to first patch only).
        zorder: Drawing order. Default 2.
        **kwargs: Forwarded to PathPatch.

    Returns:
        List of PathPatch objects added to the axis.
    """
    from matplotlib.patches import PathPatch

    ys = np.atleast_1d(y).astype(float)
    ws = np.atleast_1d(widths).astype(float)
    if len(ys) != len(ws):
        raise ValueError("y and widths must have the same length")

    pts = []
    for yi, w in zip(ys, ws):
        if w > 0:
            pts.append((0, yi - height / 2))
            pts.append((w, yi + height / 2))
    if pts:
        ax.update_datalim(pts)
        ax.autoscale_view()

    # Radius is conceptually in y-data units (proportional to bar height),
    # then converted to display pixels and back to x-data for circular
    # corners in display space.
    r_y_data = radius_frac * height
    p0 = ax.transData.transform((0, 0))
    p1 = ax.transData.transform((0, r_y_data))
    r_pixels = abs(p1[1] - p0[1])

    inv = ax.transData.inverted()
    q0 = inv.transform((0, 0))
    q1 = inv.transform((r_pixels, 0))
    r_x_data = abs(q1[0] - q0[0])

    patches = []
    for i, (yi, w) in enumerate(zip(ys, ws)):
        path = _rounded_hbar_path(float(yi), float(w), float(height),
                                   r_x_data, r_y_data)
        if path is None:
            continue
        lbl = label if (i == 0 and label is not None) else None
        patch = PathPatch(path, facecolor=color, edgecolor="none",
                          label=lbl, zorder=zorder, **kwargs)
        ax.add_patch(patch)
        patches.append(patch)

    return patches
