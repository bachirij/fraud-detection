import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import sys, os

from utils.data_loader import get_data, get_models

def render():
    st.title("Model Comparison")
    st.markdown("Performance comparison across all trained models on the test set.")
    st.markdown("---")

    # ── Load data & models ─────────────────────────────────────────────────────
    _, X_test, _, y_test, _, _ = get_data()
    models = get_models()

    # ── Compute metrics ────────────────────────────────────────────────────────
    from evaluate import compare_models
    with st.spinner("Computing metrics..."):
        df_metrics = compare_models(models, X_test, y_test)

    # ── Metrics table ──────────────────────────────────────────────────────────
    st.subheader("Metrics Summary")

    def highlight_best(s):
        """Highlight best value per column in green."""
        if s.name in ["Precision", "Recall", "F1", "AUC-ROC"]:
            is_best = s == s.max()
        elif s.name in ["FN", "FP"]:
            is_best = s == s.min()
        else:
            return [""] * len(s)
        return ["background-color: #d4edda; color: #155724" if v else "" for v in is_best]

    styled = (
        df_metrics.style
        .apply(highlight_best)
        .format({
            "Precision": "{:.3f}",
            "Recall":    "{:.3f}",
            "F1":        "{:.3f}",
            "AUC-ROC":   "{:.4f}",
            "FN":        "{:.0f}",
            "FP":        "{:.0f}",
        })
    )
    st.dataframe(styled, use_container_width=True)

    st.markdown("---")

    # ── ROC curves ────────────────────────────────────────────────────────────
    st.subheader("ROC Curves")

    from sklearn.metrics import roc_curve, auc

    fig = go.Figure()

    colors = {
        "Logistic Regression": "steelblue",
        "Decision Tree":       "orange",
        "Random Forest":       "green",
        "XGBoost":             "crimson",
    }

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc_score = auc(fpr, tpr)

        fig.add_trace(go.Scatter(
            x=fpr, y=tpr,
            mode="lines",
            name=f"{model_name} (AUC = {auc_score:.4f})",
            line=dict(color=colors.get(model_name, "gray"), width=2)
        ))

    # Random baseline
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode="lines",
        name="Random baseline",
        line=dict(color="gray", width=1, dash="dash")
    ))

    fig.update_layout(
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        legend=dict(x=0.6, y=0.1),
        height=500
    )

    st.plotly_chart(fig, use_container_width=True)