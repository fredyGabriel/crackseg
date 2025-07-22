"""
Setup guide section for the results page. This module handles the
setup guide displayed when users haven't completed the necessary
prerequisites for viewing results.
"""

import streamlit as st


def render_setup_guide() -> None:
    """Render setup guide for users who haven't completed training."""
    st.warning("**Setup Required**")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
### **Get Started** To access results visualization, you need: 1.
**Complete Training** - Train a model first 2. **Set Run Directory** -
Configure output directory 3. **Generate Predictions** - Run inference
on test data
"""
        )

    with col2:
        st.markdown(
            """
### **Quick Actions** - **[Training Page](/training)** - Start model
training - **[Configuration](/config)** - Set up directories -
**[Documentation](docs/)** - View training guide
"""
        )

    st.info(
        """
**Tip**: Once you have completed training, return to this page to: -
View TensorBoard metrics - Browse prediction triplets (Image | Mask |
Prediction) - Export results in various formats - Compare model
performance
"""
    )
