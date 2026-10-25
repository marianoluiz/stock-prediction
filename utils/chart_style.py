"""Shared matplotlib style for every figure the thesis scripts write.

One look across all charts: white surface, recessive hairline grid on the
value axis only, no top/right spines, frameless legend above the plot, and
text in ink tones (never in a series color). Series colors are fixed by role
so a model keeps its color in every figure:

* MSE baseline       -- blue   (categorical slot 1)
* Profit-aware       -- orange (categorical slot 2)
* third condition    -- aqua   (categorical slot 3; only used where a figure
                        has a third model variant)
* passive reference  -- gray   (buy & hold, classic strategies, index)

Slots 1-3 pass the colorblind/normal-vision separation checks as a set. Aqua
sits below 3:1 contrast on white, so any figure using it also writes its
numbers to a CSV/summary file alongside.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import PercentFormatter

SURFACE = "#ffffff"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"

COLOR_MSE = "#2a78d6"
COLOR_PA = "#eb6834"
COLOR_THIRD = "#1baf7a"
COLOR_REF = "#898781"

LABEL_MSE = "MSE baseline"
LABEL_PA = "Profit-aware"


def apply_style() -> None:
    """Install the shared rcParams. Safe to call more than once."""
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": "sans-serif",
        "font.sans-serif": ["Segoe UI", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.titlesize": 11,
        "axes.titleweight": "semibold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": GRID,
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "grid.linestyle": "-",
        "xtick.color": INK_SECONDARY,
        "ytick.color": INK_SECONDARY,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "legend.frameon": False,
        "legend.fontsize": 9,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def title_block(ax, title: str, subtitle: str | None = None) -> None:
    """Left-aligned title, then an optional subtitle, stacked above the plot.

    Call after :func:`legend_above`: the block is placed above the legend so
    the three never overlap (title / subtitle / legend / plot, top to bottom).
    """
    top = 1.0
    legend = ax.get_legend()
    if legend is not None:
        ax.figure.canvas.draw()
        top = legend.get_window_extent().transformed(ax.transAxes.inverted()).y1
    offset = 4
    if subtitle:
        ax.annotate(subtitle, (0.0, top), xycoords="axes fraction", xytext=(0, offset),
                    textcoords="offset points", ha="left", va="bottom", fontsize=8.5, color=INK_SECONDARY)
        offset += 15
    ax.annotate(title, (0.0, top), xycoords="axes fraction", xytext=(0, offset), textcoords="offset points",
                ha="left", va="bottom", fontsize=11, fontweight="semibold", color=INK)


def legend_above(ax, ncol: int | None = None, handles=None, labels=None) -> None:
    """Frameless one-row legend directly above the plot, aligned to its left edge.

    Data series (bars, filled marks) are listed before reference lines.
    """
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
        pairs = sorted(zip(handles, labels), key=lambda hl: isinstance(hl[0], Line2D) and hl[0].get_linestyle() != "-")
        handles, labels = [h for h, _ in pairs], [lbl for _, lbl in pairs]
    ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 1.0),
              ncol=ncol or len(labels), handlelength=1.2, columnspacing=1.4, borderaxespad=0.3)


def zero_line(ax, axis: str = "y") -> None:
    """Solid baseline at zero in the axis color, so bars read up/down from it."""
    (ax.axhline if axis == "y" else ax.axvline)(0, color=INK_MUTED, linewidth=0.9, zorder=2)


def percent_axis(ax, axis: str = "y", decimals: int = 0) -> None:
    """Format a fraction-valued axis as percent (0.12 -> 12%)."""
    fmt = PercentFormatter(xmax=1.0, decimals=decimals)
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)


def label_bars(ax, bars, fmt: str, horizontal: bool = False, fontsize: float = 7.5) -> None:
    """Value at the tip of each bar, in secondary ink, outside the bar end."""
    for bar in bars:
        if horizontal:
            v = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.annotate(fmt.format(v), (v, y), xytext=(3 if v >= 0 else -3, 0), textcoords="offset points",
                        ha="left" if v >= 0 else "right", va="center", fontsize=fontsize, color=INK_SECONDARY)
        else:
            v = bar.get_height()
            x = bar.get_x() + bar.get_width() / 2
            ax.annotate(fmt.format(v), (x, v), xytext=(0, 2 if v >= 0 else -2), textcoords="offset points",
                        ha="center", va="bottom" if v >= 0 else "top", fontsize=fontsize, color=INK_SECONDARY)


def footnote(fig, text: str) -> None:
    """Muted source/assumption line at the bottom-left of the figure."""
    fig.text(0.01, 0.005, text, ha="left", va="bottom", fontsize=7.5, color=INK_MUTED)


def save(fig, path: Path, dpi: int = 200) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path
