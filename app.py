import streamlit as st
from regression import regression_section

# =============================
# App Configuration
# =============================
st.set_page_config(
    page_title="📈 Regression Analysis",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================
# Sidebar Navigation
# =============================
st.sidebar.title("📊 Regression Explorer")
st.sidebar.markdown("---")

# Define sidebar options
menu_options = {"🏠 Home": "Home", "📈 Regression": "Regression"}

# Use Streamlit-native radio button for navigation
section = st.sidebar.radio("🔍 Navigate to:", list(menu_options.keys()))

# Highlight selected section title in the sidebar header
if section == "🏠 Home":
    st.sidebar.success("Home Page")
elif section == "📈 Regression":
    st.sidebar.success("Regression Page")

st.sidebar.markdown("---")
st.sidebar.markdown("👤 **By:** TerraTech")
st.sidebar.markdown("🎓 **Datathon Group 08**")
st.sidebar.markdown("📅 **Presented for:** AI/ML Datathon 2025")
st.sidebar.markdown("---")

# =============================
# HOME PAGE
# =============================
if section == "🏠 Home":
    st.markdown(
        "<h1 style='text-align: center;'>🎯 Regression Analysis Demo</h1>",
        unsafe_allow_html=True,
    )

    with st.expander("ℹ️ About This App", expanded=True):
        st.markdown(
            """
        Welcome to the **Regression Analysis Demo App**!  
        This project demonstrates how machine learning models can predict continuous outcomes
        using regression techniques.

        🔍 **You can:**
        - Upload a dataset (CSV)
        - Select your features and target column
        - Compare multiple regression models
        - Tune hyperparameters interactively
        - Visualize model performance
        """
        )

    st.subheader("📘 Instructions")
    st.markdown(
        """
    1. Navigate to the **Regression** page using the sidebar.  
    2. Upload your dataset (e.g., housing prices, crop yield, etc.).  
    3. Select the target and feature columns.  
    4. Train and compare multiple regression models.  
    5. Tune hyperparameters and visualize predictions.  
    """
    )

# =============================
# REGRESSION PAGE
# =============================
elif section == "📈 Regression":
    regression_section()
