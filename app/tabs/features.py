import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data_loader import get_models

def render():
    st.title("Feature Importance")
    st.markdown("Which features drive fraud detection the most, across models.")
    st.markdown("---")

    # ── Load models ────────────────────────────────────────────────────────────
    models = get_models()
    xgb_model = models["XGBoost"]
    rf_model  = models["Random Forest"]

    feature_names = [f"V{i}" for i in range(1, 29)] + ["Amount", "Time"]

    # ── Extract importances ────────────────────────────────────────────────────
    xgb_imp = pd.Series(xgb_model.feature_importances_, index=feature_names)
    rf_imp  = pd.Series(rf_model.feature_importances_,  index=feature_names)

    xgb_top = xgb_imp.sort_values(ascending=False).head(15)
    rf_top  = rf_imp.sort_values(ascending=False).head(15)

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("XGBoost - Top 15 Features")
        fig = go.Figure(go.Bar(
            x=xgb_top.values,
            y=xgb_top.index,
            orientation="h",
            marker_color="crimson",
            opacity=0.8,
        ))
        fig.update_layout(
            xaxis_title="Importance score",
            yaxis=dict(autorange="reversed"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Random Forest - Top 15 Features")
        fig = go.Figure(go.Bar(
            x=rf_top.values,
            y=rf_top.index,
            orientation="h",
            marker_color="steelblue",
            opacity=0.8,
        ))
        fig.update_layout(
            xaxis_title="Importance score",
            yaxis=dict(autorange="reversed"),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Common top features insight ────────────────────────────────────────────
    st.subheader("Consensus Features")
    st.markdown("Features ranked in the **top 10 of both models** - highest confidence signals.")

    xgb_top10 = set(xgb_imp.sort_values(ascending=False).head(10).index)
    rf_top10  = set(rf_imp.sort_values(ascending=False).head(10).index)
    common    = sorted(xgb_top10 & rf_top10)

    if common:
        cols = st.columns(len(common))
        for col, feat in zip(cols, common):
            xgb_rank = list(xgb_imp.sort_values(ascending=False).index).index(feat) + 1
            rf_rank  = list(rf_imp.sort_values(ascending=False).index).index(feat) + 1
            col.metric(
                label=feat,
                value=f"XGB #{xgb_rank}",
                delta=f"RF #{rf_rank}",
                delta_color="off"
            )
    else:
        st.info("No features in common in top 10.")