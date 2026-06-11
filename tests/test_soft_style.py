"""Unit tests for the soft_style helpers."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pytest

import soft_style as ss


@pytest.fixture(autouse=True)
def _style():
    ss.configure_style()
    yield
    plt.close("all")
    ss.configure_style()  # reset any scale a test applied


# ------------------------------------------------------- configure_style --

def test_pdf_embeds_truetype_fonts():
    # Type 3 fonts are rejected by IEEE/ACM/NeurIPS checkers
    assert plt.rcParams["pdf.fonttype"] == 42
    assert plt.rcParams["ps.fonttype"] == 42


def test_scale_multiplies_font_sizes():
    ss.configure_style(scale=0.75)
    assert plt.rcParams["font.size"] == pytest.approx(11 * 0.75)
    assert plt.rcParams["figure.titlesize"] == pytest.approx(18 * 0.75)
    assert plt.rcParams["lines.linewidth"] == pytest.approx(2.0 * 0.75)


def test_cream_background():
    ss.configure_style(cream_bg=True)
    assert plt.rcParams["figure.facecolor"] == ss.NEUTRAL["cream_bg"]
    ss.configure_style()
    assert plt.rcParams["figure.facecolor"] == "white"


# --------------------------------------------------------- rounded bars --

def test_rounded_bars_linear():
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0, 1)
    patches = ss.rounded_bars(ax, [0, 1, 2], [0.5, 0.8, 0.3], width=0.6,
                              color="#D8704C")
    assert len(patches) == 3


def test_rounded_bars_log_y():
    # Regression guard for the PR #3 log-axis fix
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(1e-2, 1e2)
    patches = ss.rounded_bars(ax, [0, 1, 2], [10.0, 50.0, 0.5], width=0.6,
                              color="#D8704C")
    assert len(patches) == 3
    for p in patches:
        assert np.all(np.isfinite(p.get_path().vertices))


def test_rounded_bars_stacking_baseline_array():
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 2)
    bottom = np.array([0.5, 0.8])
    patches = ss.rounded_bars(ax, [0, 1], bottom + np.array([0.6, 0.4]),
                              width=0.5, color="#5C5B58", baseline=bottom)
    assert len(patches) == 2
    # each patch must start at its own baseline, not at 0
    for p, b in zip(patches, bottom):
        assert p.get_path().vertices[:, 1].min() == pytest.approx(b, abs=1e-6)


def test_rounded_bars_skips_degenerate():
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 1)
    patches = ss.rounded_bars(ax, [0, 1], [0.0, 0.5], width=0.5,
                              color="#D8704C")
    assert len(patches) == 1  # zero-height bar skipped


def test_rounded_bars_length_mismatch():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ss.rounded_bars(ax, [0, 1], [1.0], width=0.5, color="#D8704C")


def test_rounded_hbars():
    fig, ax = plt.subplots()
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 2.5)
    patches = ss.rounded_hbars(ax, [0, 1, 2], [0.3, 0.9, 0.6], height=0.6,
                               color="#D8704C")
    assert len(patches) == 3


# -------------------------------------------------------------- helpers --

def test_top_legend_on_axes():
    from matplotlib.patches import Patch
    fig, ax = plt.subplots()
    handles = [Patch(facecolor="#D8704C", label="a"),
               Patch(facecolor="#EBC5A8", label="b")]
    leg = ss.top_legend(ax, handles)
    assert leg.get_frame_on() is False
    assert len(leg.get_texts()) == 2


def test_top_legend_on_figure():
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2)
    handles = [Line2D([0], [0], color="#6A9CC9", label="x")]
    leg = ss.top_legend(fig, handles)
    assert leg in fig.legends


def test_plain_log_ticks():
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    ax.set_ylim(1.5, 10)
    ss.plain_log_ticks(ax, [2, 3, 4, 6, 9])
    fig.canvas.draw()
    labels = [t.get_text() for t in ax.get_yticklabels()]
    assert labels == ["2", "3", "4", "6", "9"]
    assert all(t.get_text() == "" for t in ax.get_yticklabels(minor=True))


def test_plain_log_ticks_bad_axis():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ss.plain_log_ticks(ax, [1, 2], axis="z")


def test_soft_colorbar_from_mappable():
    fig, ax = plt.subplots()
    im = ax.imshow(np.eye(3), cmap=ss.CMAP_SEQUENTIAL)
    cbar = ss.soft_colorbar(fig, ax, im, label="value")
    assert cbar.ax.get_ylabel() == "value"


def test_soft_colorbar_from_norm_cmap():
    fig, ax = plt.subplots()
    norm = mpl.colors.LogNorm(1e5, 1e11)
    cbar = ss.soft_colorbar(fig, ax, norm=norm, cmap=ss.CMAP_GRADIENT,
                            label="params")
    assert cbar.ax.get_ylabel() == "params"


def test_soft_colorbar_requires_inputs():
    fig, ax = plt.subplots()
    with pytest.raises(ValueError):
        ss.soft_colorbar(fig, ax)


# ------------------------------------------------------------ smoothing --

def test_smooth_curve_shapes():
    x = np.linspace(0, 10, 50)
    y = np.sin(x)
    xs, ys = ss.smooth_curve(x, y, frac=0.3)
    assert len(xs) == len(ys) == 200


def test_rolling_band_ordering():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 10, 80)
    y = np.sin(x) + rng.normal(0, 0.1, 80)
    xs, lo, hi, mean = ss.rolling_band(x, y, frac=0.3, k=1.0)
    assert np.all(lo <= mean) and np.all(mean <= hi)


# -------------------------------------------------------------- export --

def test_save_figure_writes_files(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fig, ax = plt.subplots()
    ax.plot([0, 1], [0, 1])
    ss.save_figure(fig, "out", formats=("png", "pdf"), dpi=72)
    assert (tmp_path / "out.png").exists()
    assert (tmp_path / "out.pdf").exists()
