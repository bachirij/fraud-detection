import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from utils.data_loader import get_models, get_scaler, get_data

def render():
    st.title("Transaction Scorer")
    st.markdown("Score an individual transaction using the XGBoost model.")
    st.markdown("---")

    models = get_models()
    model  = models["XGBoost"]
    scaler = get_scaler()

    _, X_test, _, y_test, _, _ = get_data()
    y_test_arr = y_test.values if hasattr(y_test, "values") else y_test

    # ── Quick test buttons ─────────────────────────────────────────────────────
    col_demo1, col_demo2, _ = st.columns([1, 1, 3])

    with col_demo1:
        if st.button("Load random fraud"):
            fraud_indices = np.where(y_test_arr == 1)[0]
            idx = np.random.choice(fraud_indices)
            st.session_state["demo_row"]   = X_test.iloc[idx].to_dict()
            st.session_state["demo_label"] = "fraud"
            st.rerun()

    with col_demo2:
        if st.button("Load random legitimate"):
            legit_indices = np.where(y_test_arr == 0)[0]
            idx = np.random.choice(legit_indices)
            st.session_state["demo_row"]   = X_test.iloc[idx].to_dict()
            st.session_state["demo_label"] = "legit"
            st.rerun()

    if "demo_label" in st.session_state:
        if st.session_state["demo_label"] == "fraud":
            st.warning("Loaded a known fraud transaction from test set")
        else:
            st.info("ℹLoaded a known legitimate transaction from test set")

    # ── Sync session_state keys with demo_row ──────────────────────────────────
    if "demo_row" in st.session_state:
        demo_sync = st.session_state["demo_row"]
        for i in range(1, 29):
            st.session_state[f"v{i}"] = float(demo_sync.get(f"V{i}", 0.0))

    st.markdown("---")

    # ── Input form ─────────────────────────────────────────────────────────────
    st.subheader("Transaction Input")
    st.caption("Enter raw transaction values — scaling is applied automatically.")

    demo = st.session_state.get("demo_row", {})

    col1, col2 = st.columns(2)
    with col1:
        time   = st.number_input("Time (seconds since first transaction)",
                                  value=float(demo.get("Time", 50000.0)))
    with col2:
        amount = st.number_input("Amount (€)",
                                  value=float(demo.get("Amount", 50.0)))

    st.markdown("**PCA Features (V1 - V28)**")
    v_cols = st.columns(7)
    v_values = {}
    for i in range(1, 29):
        col_idx = (i - 1) % 7
        with v_cols[col_idx]:
            v_values[f"V{i}"] = st.number_input(
                f"V{i}",
                value=float(demo.get(f"V{i}", 0.0)),
                format="%.4f",
                key=f"v{i}"
            )

    st.markdown("---")

    # ── Score button ───────────────────────────────────────────────────────────
    if st.button("Score Transaction", type="primary"):

        row = {"Time": time, "Amount": amount}
        row.update(v_values)
        df_input = pd.DataFrame([row])

        df_scaled = df_input.copy()
        df_scaled[["Amount", "Time"]] = scaler.transform(df_input[["Amount", "Time"]])

        feature_cols = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]
        df_scaled = df_scaled[feature_cols]

        proba = model.predict_proba(df_scaled)[0][1]
        pred  = int(proba >= 0.30)

        st.markdown("---")
        st.subheader("Result")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(proba * 100, 2),
            number={"suffix": "%"},
            title={"text": "Fraud Probability"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar":  {"color": "crimson" if proba >= 0.30 else "steelblue"},
                "steps": [
                    {"range": [0, 30],   "color": "#d4edda"},
                    {"range": [30, 60],  "color": "#fff3cd"},
                    {"range": [60, 100], "color": "#f8d7da"},
                ],
                "threshold": {
                    "line":      {"color": "orange", "width": 3},
                    "thickness": 0.75,
                    "value":     30
                }
            }
        ))
        fig.update_layout(height=350)
        st.plotly_chart(fig, use_container_width=True)

        if pred == 1:
            st.error(f"🚨 **FRAUD DETECTED** - probability {proba:.1%} (threshold 0.30)")
        else:
            st.success(f"✅ **LEGITIMATE** - probability {proba:.1%} (threshold 0.30)")

        st.markdown("---")
        st.subheader("Top Contributing Features")
        st.caption("Feature importance × |input value| - approximate signal strength.")

        importances  = pd.Series(model.feature_importances_, index=feature_cols)
        input_vals   = df_scaled.iloc[0]
        contribution = (importances * input_vals.abs()).sort_values(ascending=False).head(10)

        fig2 = go.Figure(go.Bar(
            x=contribution.values,
            y=contribution.index,
            orientation="h",
            marker_color="crimson" if pred == 1 else "steelblue",
            opacity=0.8,
        ))
        fig2.update_layout(
            xaxis_title="Contribution score",
            yaxis=dict(autorange="reversed"),
            height=400,
        )
        st.plotly_chart(fig2, use_container_width=True)