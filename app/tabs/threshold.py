import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import get_data, get_models

def render():
    st.title("Threshold Explorer")
    st.markdown("Adjust the classification threshold and observe the business impact in real time.")
    st.markdown("---")

    # ── Load data & model (XGBoost only) ───────────────────────────────────────
    _, X_test, _, y_test, _, _ = get_data()
    models = get_models()
    model = models["XGBoost"]

    y_proba = model.predict_proba(X_test)[:, 1]
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test

    # ── Precompute metrics across all thresholds ───────────────────────────────
    @st.cache_data
    def compute_threshold_curves(_y_test, _y_proba):
        from sklearn.metrics import precision_score, recall_score, f1_score
        thresholds = np.linspace(0.01, 0.99, 200)
        precisions, recalls, f1s = [], [], []
        for t in thresholds:
            y_pred = (_y_proba >= t).astype(int)
            precisions.append(precision_score(_y_test, y_pred, zero_division=0))
            recalls.append(recall_score(_y_test, y_pred, zero_division=0))
            f1s.append(f1_score(_y_test, y_pred, zero_division=0))
        return thresholds, precisions, recalls, f1s

    thresholds, precisions, recalls, f1s = compute_threshold_curves(
        y_test_arr, y_proba
    )

    # ── Slider ─────────────────────────────────────────────────────────────────
    threshold = st.slider(
        "Classification threshold",
        min_value=0.01, max_value=0.99,
        value=0.30, step=0.01,
        help="Transactions with fraud probability ≥ threshold are flagged as fraud."
    )

    # ── Metrics at chosen threshold ────────────────────────────────────────────
    from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

    y_pred = (y_proba >= threshold).astype(int)
    precision = precision_score(y_test_arr, y_pred, zero_division=0)
    recall    = recall_score(y_test_arr, y_pred, zero_division=0)
    f1        = f1_score(y_test_arr, y_pred, zero_division=0)
    cm        = confusion_matrix(y_test_arr, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # ── KPI cards ──────────────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Precision", f"{precision:.3f}")
    col2.metric("Recall",    f"{recall:.3f}")
    col3.metric("F1",        f"{f1:.3f}")
    col4.metric("False Negatives (missed fraud)", int(fn))
    col5.metric("False Positives (blocked legit)", int(fp))

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Precision / Recall / F1 vs threshold
    with col_left:
        st.subheader("Metrics vs Threshold")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=thresholds, y=precisions, mode="lines",
                                 name="Precision", line=dict(color="steelblue")))
        fig.add_trace(go.Scatter(x=thresholds, y=recalls, mode="lines",
                                 name="Recall", line=dict(color="crimson")))
        fig.add_trace(go.Scatter(x=thresholds, y=f1s, mode="lines",
                                 name="F1", line=dict(color="green", dash="dash")))
        fig.add_vline(x=threshold, line_dash="dot", line_color="orange",
                      annotation_text=f"t={threshold:.2f}")
        fig.update_layout(xaxis_title="Threshold", yaxis_title="Score", height=400)
        st.plotly_chart(fig, use_container_width=True)

    # Confusion matrix
    with col_right:
        st.subheader("Confusion Matrix")
        z = [[tn, fp], [fn, tp]]
        text = [[f"TN\n{tn:,}", f"FP\n{fp:,}"],
                [f"FN\n{fn:,}", f"TP\n{tp:,}"]]
        fig = go.Figure(go.Heatmap(
            z=z, text=text, texttemplate="%{text}",
            colorscale="Blues", showscale=False,
            x=["Predicted Legit", "Predicted Fraud"],
            y=["Actual Legit", "Actual Fraud"],
        ))
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")

    # ── Business impact ────────────────────────────────────────────────────────
    st.subheader("Business Impact Estimation")

    # Reconstruct a clean DataFrame with aligned indices
    impact_df = X_test.copy()
    impact_df["true_label"] = y_test_arr
    impact_df["predicted"]  = y_pred

    missed = impact_df[(impact_df["true_label"] == 1) & (impact_df["predicted"] == 0)]
    missed_exposure = missed["Amount"].sum()
    fn_count = len(missed)

    blocked = impact_df[(impact_df["true_label"] == 0) & (impact_df["predicted"] == 1)]
    fp_count = len(blocked)

    avg_fraud = impact_df[impact_df["true_label"] == 1]["Amount"].mean()

    col_a, col_b = st.columns(2)
    with col_a:
        st.error(f"🚨 Missed fraud: **{fn_count} transactions** - exposure **€{missed_exposure:,.0f}**")
    with col_b:
        st.warning(f"⚠️ Blocked legitimate: **{fp_count} transactions** flagged incorrectly")

    st.caption(f"Average fraud amount in test set: €{avg_fraud:.2f}")