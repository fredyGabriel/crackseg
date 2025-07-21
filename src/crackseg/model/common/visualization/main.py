"""Main visualization module for U-Net architecture diagrams.

This module provides the main entry point for rendering U-Net architecture
diagrams with automatic backend selection and fallback options.
"""

import logging
from typing import Any

from .graphviz_renderer import render_unet_architecture_graphviz
from .matplotlib_renderer import render_unet_architecture_matplotlib

logger = logging.getLogger(__name__)


def render_unet_architecture_diagram(
    layer_hierarchy: list[dict[str, Any]],
    filename: str | None = None,
    view: bool = False,
    backend: str = "matplotlib",
    figsize: tuple[float, float] = (12, 8),
) -> None:
    """Render a U-Net architecture diagram using matplotlib or graphviz.

    Creates a U-shaped block diagram showing the model architecture with
    component relationships, skip connections, and data flow. Uses matplotlib
    by default for better compatibility (ADR-001), with graphviz as fallback.

    Args:
        layer_hierarchy: Model layer information from get_layer_hierarchy()
        filename: Optional output filename for saving diagram.
            If None, generates temporary file or shows plot.
            Supports formats: .png, .pdf, .svg, .jpg
        view: If True, automatically displays the plot.
            Uses backend-specific display method.
        backend: Visualization backend to use. Options:
            - "matplotlib" (default): Uses matplotlib for rendering
            - "graphviz": Uses graphviz for rendering (if available)
            - "auto": Try matplotlib first, fallback to graphviz
        figsize: Figure size for matplotlib backend as (width, height) tuple.
            Ignored for graphviz backend.

    Examples:
        >>> # Use matplotlib (default)
        >>> hierarchy = get_layer_hierarchy(encoder, bottleneck, decoder)
        >>> render_unet_architecture_diagram(hierarchy, "arch.png", view=True)

        >>> # Force graphviz if available
        >>> render_unet_architecture_diagram(
        ...     hierarchy, "arch.png", backend="graphviz"
        ... )

        >>> # Auto-select backend
        >>> render_unet_architecture_diagram(
        ...     hierarchy, "arch.png", backend="auto"
        ... )

    Notes:
        - Matplotlib backend is preferred for reliability (ADR-001)
        - Graphviz backend requires additional system dependencies
        - Auto backend tries matplotlib first, then graphviz
        - Both backends produce visually similar U-Net diagrams
    """
    if backend == "matplotlib":
        try:
            render_unet_architecture_matplotlib(
                layer_hierarchy, filename, view, figsize
            )
            return
        except ImportError as e:
            logger.warning(f"Matplotlib backend failed: {e}")
            if backend == "matplotlib":  # Strict matplotlib mode
                raise

    elif backend == "graphviz":
        try:
            render_unet_architecture_graphviz(layer_hierarchy, filename, view)
            return
        except ImportError as e:
            logger.warning(f"Graphviz backend failed: {e}")
            if backend == "graphviz":  # Strict graphviz mode
                raise

    elif backend == "auto":
        # Try matplotlib first (preferred)
        try:
            render_unet_architecture_matplotlib(
                layer_hierarchy, filename, view, figsize
            )
            return
        except ImportError:
            logger.info("Matplotlib not available, trying graphviz...")

        # Fallback to graphviz
        try:
            render_unet_architecture_graphviz(layer_hierarchy, filename, view)
            return
        except ImportError:
            pass

    # If we get here, no backend worked
    raise ImportError(
        f"No visualization backend available. Tried: {backend}. "
        "Install matplotlib (recommended) or graphviz for architecture "
        "visualization."
    )
