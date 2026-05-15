import sys
import os

# ── Path setup (must be before all other imports) ──────────────────────────────
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.append(os.path.dirname(__file__))  # adds app/ for utils.data_loader

import streamlit as st
from tabs.overview import render as render_overview
from tabs.comparison import render as render_comparison
from tabs.threshold import render as render_threshold
from tabs.features import render as render_features
from tabs.scorer import render as render_scorer

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar navigation ─────────────────────────────────────────────────────────
st.sidebar.title("🔍 Fraud Detection")
st.sidebar.markdown("---")

PAGES = {
    "Business Overview":      "overview",
    "Model Comparison":       "comparison",
    "Threshold Explorer":     "threshold",
    "Feature Importance":     "features",
    "Transaction Scorer":     "scorer",
}

selection = st.sidebar.radio("Navigation", list(PAGES.keys()))
page = PAGES[selection]

st.sidebar.markdown("---")
st.sidebar.caption("Credit Card Fraud Detection · ULB Dataset")

# ── Page routing ───────────────────────────────────────────────────────────────
if page == "overview":
    render_overview()

elif page == "comparison":
    render_comparison()

elif page == "threshold":
    render_threshold()

elif page == "features":
    render_features()

elif page == "scorer":
    render_scorer()