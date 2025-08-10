"""Figure utilities for publication-ready reporting.

Provides helpers to configure matplotlib style and to save figures in
multiple formats. Designed to accept a generic ``style`` object with
attributes used by publication figure generators, avoiding tight coupling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.figure import Figure


def setup_publication_style(style: Any) -> None:
    """Configure matplotlib/seaborn for publication-quality figures.

    Expected attributes on ``style``: dpi, font_family, font_size,
    title_font_size, legend_font_size, line_width, marker_size, grid_alpha,
    color_palette, tight_layout, transparent_background.
    """
    plt.style.use("seaborn-v0_8-paper")
    plt.rcParams.update(
        {
            "figure.dpi": getattr(style, "dpi", 300),
            "font.family": getattr(style, "font_family", "serif"),
            "font.size": getattr(style, "font_size", 10),
            "axes.titlesize": getattr(style, "title_font_size", 12),
            "axes.labelsize": getattr(style, "font_size", 10),
            "xtick.labelsize": max(1, getattr(style, "font_size", 10) - 1),
            "ytick.labelsize": max(1, getattr(style, "font_size", 10) - 1),
            "legend.fontsize": getattr(style, "legend_font_size", 9),
            "lines.linewidth": getattr(style, "line_width", 1.5),
            "lines.markersize": getattr(style, "marker_size", 6.0),
            "grid.alpha": getattr(style, "grid_alpha", 0.3),
            "savefig.dpi": getattr(style, "dpi", 300),
            "savefig.bbox": (
                "tight" if getattr(style, "tight_layout", True) else "standard"
            ),
            "savefig.transparent": getattr(
                style, "transparent_background", True
            ),
        }
    )
    sns.set_palette(getattr(style, "color_palette", "viridis"))


def save_figure_multiple_formats(
    fig: Figure, base_path: Path, formats: list[str] | None, style: Any
) -> dict[str, Path]:
    """Save ``fig`` under ``base_path`` in given ``formats``.

    Uses ``style.supported_formats`` as allowlist when available.
    """
    saved: dict[str, Path] = {}
    formats = formats or getattr(style, "supported_formats", None)
    allowlist = getattr(style, "supported_formats", None)
    if not formats:
        return saved

    for fmt in formats:
        if allowlist is not None and fmt not in allowlist:
            continue
        path = base_path.with_suffix(f".{fmt}")
        try:
            if fmt == "png":
                fig.savefig(
                    path, dpi=getattr(style, "dpi", 300), bbox_inches="tight"
                )
            elif fmt == "svg":
                fig.savefig(path, format="svg", bbox_inches="tight")
            elif fmt == "pdf":
                fig.savefig(path, format="pdf", bbox_inches="tight")
            else:
                continue
            saved[fmt] = path
        except Exception:
            # Do not raise to avoid breaking pipelines on an optional format
            continue

    return saved
