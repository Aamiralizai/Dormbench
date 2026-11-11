"""Shared figure theme. All project figures are SVG."""

from __future__ import annotations

from pathlib import Path

INK, MUTED, GRID = "#1a1a2e", "#7a7a8c", "#e4e4ec"
PALETTE = ["#1b6ca8", "#c1272d", "#2a9d8f", "#e9c46a", "#8e6bb5", "#6b7280"]


def apply_theme() -> None:
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "svg.fonttype": "none",          # keep text as text, editable in Inkscape
        "axes.edgecolor": GRID,
        "axes.labelcolor": INK,
        "axes.titlesize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "grid.color": GRID,
        "grid.linewidth": 0.7,
        "figure.dpi": 110,
    })


def save_svg(fig, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, format="svg", bbox_inches="tight")
    return path
