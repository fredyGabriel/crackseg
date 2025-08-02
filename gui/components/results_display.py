"""
This module defines the TripletDisplayComponent for the results gallery.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import streamlit as st
from PIL import Image
from streamlit.delta_generator import DeltaGenerator

from gui.utils.results import ResultTriplet, TripletHealth


class TripletDisplayComponent:
    """
    A Streamlit component to display result triplets (image, mask, prediction)
    in a responsive and interactive grid.
    """

    def __init__(
        self,
        triplets: list[ResultTriplet],
        selected_ids: set[str],
        on_selection_change: Callable[[str], None],
        on_batch_selection_change: Callable[[set[str], bool], None],
        items_per_page: int = 12,
    ):
        """
        Initializes the component with a list of result triplets.

        Args:
            triplets (list[ResultTriplet]): The list of triplets to display.
            selected_ids (set[str]): A set of the currently selected triplet
            IDs.
            on_selection_change (Callable[[str], None]): Callback function
                when a single triplet's selection state changes.
            on_batch_selection_change (Callable[[set[str], bool], None]):
                Callback for batch selection changes (add/remove).
            items_per_page (int, optional): Number of items per page. Defaults
                to 12.
        """
        self.triplets = triplets
        self.selected_ids = selected_ids
        self.on_selection_change = on_selection_change
        self.on_batch_selection_change = on_batch_selection_change
        self.items_per_page = items_per_page
        self._init_state()

    def _init_state(self) -> None:
        """Initializes the session state required for this component."""
        if "results_display_page" not in st.session_state:
            st.session_state.results_display_page = 0
        if "selected_triplet" not in st.session_state:
            st.session_state.selected_triplet = None

    def render(self, container: DeltaGenerator | None = None) -> None:
        """
        Renders the triplet display component.

        This method will handle the layout, pagination, and interactivity.

        Args:
            container (DeltaGenerator | None, optional): The container
            to render the component in. Defaults to st.container().
        """
        if container is None:
            container = st.container()

        if container is not None:
            with container:
                st.write(f"Displaying {len(self.triplets)} result triplets.")
                # --- Phase 2: Basic Layout will be implemented here ---
                # This will include pagination and the grid display.

                # --- Phase 3: Core Interactivity will be implemented here ---
                # This will handle the modal view for detailed inspection.
                self._render_grid()
                self._render_detail_view()

    def _render_grid(self) -> None:
        """Renders the grid of triplet thumbnails with pagination."""
        if not self.triplets:
            st.info("No results found to display.", icon="📂")
            return

        total_pages = (
            math.ceil(len(self.triplets) / self.items_per_page)
            if self.triplets
            else 1
        )
        current_page = st.session_state.results_display_page

        start_index = current_page * self.items_per_page
        end_index = min(start_index + self.items_per_page, len(self.triplets))
        page_triplets = self.triplets[start_index:end_index]

        # --- Selection Controls ---
        st.markdown("#### Select Results for Export")
        page_triplet_ids = {t.id for t in page_triplets}
        col_sel1, col_sel2, _ = st.columns([2, 2, 8])
        if col_sel1.button("Select All on Page", use_container_width=True):
            self.on_batch_selection_change(page_triplet_ids, True)
            st.rerun()

        if col_sel2.button("Deselect All on Page", use_container_width=True):
            self.on_batch_selection_change(page_triplet_ids, False)
            st.rerun()

        st.caption(
            f"{len(self.selected_ids)} of "
            f"{len(self.triplets)} triplets selected."
        )
        st.divider()

        # --- Grid Display ---
        num_columns = 4
        cols = st.columns(num_columns)
        for i, triplet in enumerate(page_triplets):
            col = cols[i % num_columns]
            with col.container(border=True):
                health_icon = ""
                if triplet.health_status == TripletHealth.DEGRADED:
                    health_icon = "⚠️"
                elif triplet.health_status == TripletHealth.BROKEN:
                    health_icon = "❌"

                # Display image or placeholder if missing
                if triplet.image_path and triplet.image_path.exists():
                    image = Image.open(triplet.image_path)
                    st.image(
                        image,
                        caption=f"{triplet.dataset_name} | {triplet.id}",
                        use_column_width=True,
                    )
                else:
                    st.warning(f"Image for {triplet.id} is missing.", icon="🖼️")

                c1, c2 = st.columns([1, 4])
                with c1:
                    st.checkbox(
                        "",
                        value=triplet.id in self.selected_ids,
                        key=f"select_{triplet.id}",
                        on_change=self.on_selection_change,
                        args=(triplet.id,),
                        disabled=triplet.health_status == TripletHealth.BROKEN,
                        label_visibility="collapsed",
                    )
                with c2:
                    if st.button(
                        f"{health_icon} View Details",
                        key=f"details_{triplet.id}",
                        use_container_width=True,
                    ):
                        st.session_state.selected_triplet = triplet
                        st.rerun()

        st.divider()

        if total_pages > 1:
            col1, col2, col3 = st.columns([2, 6, 2])
            with col1:
                if st.button("⬅️ Previous", disabled=(current_page == 0)):
                    st.session_state.results_display_page -= 1
                    st.rerun()
            with col3:
                if st.button(
                    "Next ➡️", disabled=(current_page >= total_pages - 1)
                ):
                    st.session_state.results_display_page += 1
                    st.rerun()
            with col2:
                # Page jumper
                page_jump = st.number_input(
                    "Page",
                    min_value=1,
                    max_value=total_pages,
                    value=current_page + 1,
                    label_visibility="collapsed",
                    key="page_jumper",
                )
                if page_jump != current_page + 1:
                    st.session_state.results_display_page = page_jump - 1
                    st.rerun()

                st.markdown(
                    f"<div style='text-align: center;'>Page {current_page + 1}"
                    f" of {total_pages}</div>",
                    unsafe_allow_html=True,
                )

    def _render_detail_view(self) -> None:
        """Renders the detailed modal view for a selected triplet."""
        if (
            "selected_triplet" not in st.session_state
            or not st.session_state.selected_triplet
        ):
            return

        triplet = st.session_state.selected_triplet
        with st.dialog("Triplet Detail"):  # type: ignore
            st.subheader(f"Details for: {triplet.id}")

            if triplet.health_status != TripletHealth.HEALTHY:
                st.warning(
                    f"**Status:** {triplet.health_status.value} "
                    f"({len(triplet.missing_files)} file(s) missing)",
                    icon="⚠️",
                )
                with st.expander("Missing File Details"):
                    for file in triplet.missing_files:
                        st.code(str(file), language=None)

            self._render_image_comparison(triplet)
            if st.button("Close", key=f"close_dialog_{triplet.id}"):
                st.session_state.selected_triplet = None
                st.rerun()

    def _render_image_comparison(self, triplet: ResultTriplet) -> None:
        """Renders the interactive image comparison view."""
        image = (
            Image.open(triplet.image_path).convert("RGBA")
            if triplet.image_path and triplet.image_path.exists()
            else None
        )
        mask = (
            Image.open(triplet.mask_path).convert("RGBA")
            if triplet.mask_path and triplet.mask_path.exists()
            else None
        )
        prediction = (
            Image.open(triplet.prediction_path).convert("RGBA")
            if triplet.prediction_path and triplet.prediction_path.exists()
            else None
        )

        if not all([image, mask, prediction]):
            st.error("Cannot render comparison due to missing files.")
            # Display what we have
            cols = st.columns(3)
            if image:
                cols[0].image(image, caption="Original Image")
            if mask:
                cols[1].image(mask, caption="Ground Truth (Mask)")
            if prediction:
                cols[2].image(prediction, caption="Prediction")
            return

        # Type guards for linter
        assert image is not None
        assert mask is not None
        assert prediction is not None

        # Create colored overlays for mask and prediction
        colored_mask = self._create_colored_overlay(
            mask, color=(255, 0, 0, 150)
        )  # Red overlay
        colored_prediction = self._create_colored_overlay(
            prediction, color=(0, 255, 0, 150)
        )  # Green overlay

        st.write(
            "Use the slider to blend the prediction over the original image."
        )
        opacity = st.slider(
            "Prediction Opacity",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            key=f"slider_{triplet.id}",
        )

        # Blend the colored prediction onto the original image
        blended_image = Image.alpha_composite(
            image,
            Image.blend(
                Image.new("RGBA", image.size, (0, 0, 0, 0)),
                colored_prediction,
                alpha=opacity,
            ),
        )

        st.image(
            blended_image,
            caption="Original vs. Prediction",
            use_column_width=True,
        )

        st.divider()

        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Original Image")
        with col2:
            st.image(colored_mask, caption="Ground Truth (Mask)")
        with col3:
            st.image(colored_prediction, caption="Model Prediction")

    @staticmethod
    def _create_colored_overlay(
        image: Image.Image, color: tuple[int, int, int, int]
    ) -> Image.Image:
        """Creates a colored overlay from a grayscale or binary image."""
        overlay = Image.new("RGBA", image.size, color)
        # Use the original image's alpha channel as the mask
        alpha_mask = image.getchannel("A")
        overlay.putalpha(alpha_mask)
        return overlay

    @staticmethod
    def create_example() -> TripletDisplayComponent:
        """Creates an example instance for demonstration."""
        # In a real scenario, this would create mock ResultTriplet objects.
        st.warning("Example creation not yet implemented.", icon="⚠️")
        return TripletDisplayComponent(
            triplets=[],
            selected_ids=set(),
            on_selection_change=lambda _: None,
            on_batch_selection_change=lambda _, __: None,
            items_per_page=4,
        )
