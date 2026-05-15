import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from utils.data_loader import get_data

def render():
    st.title("Business Overview")
    st.markdown("High-level statistics on the dataset and fraud distribution.")
    st.markdown("---")

    # ── Load data ──────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test, _, _ = get_data()

    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test])
    df = X_all.copy()
    df["Class"] = y_all.values

    # ── KPI cards ──────────────────────────────────────────────────────────────
    total        = len(df)
    n_fraud      = df["Class"].sum()
    fraud_rate   = n_fraud / total * 100
    amount_exposed = df.loc[df["Class"] == 1, "Amount"].sum()

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Transactions",  f"{total:,}")
    col2.metric("Fraudulent",          f"{int(n_fraud):,}")
    col3.metric("Fraud Rate",          f"{fraud_rate:.2f}%")
    col4.metric("Amount Exposed",      f"€{amount_exposed:,.0f}")

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────────
    col_left, col_right = st.columns(2)

    # Amount distribution
    with col_left:
        st.subheader("Amount Distribution")

        legit_amounts = df.loc[df["Class"] == 0, "Amount"].clip(upper=500)
        fraud_amounts = df.loc[df["Class"] == 1, "Amount"].clip(upper=500)

        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            subplot_titles=("Legitimate transactions", "Fraudulent transactions"),
            vertical_spacing=0.12
        )

        fig.add_trace(
            go.Histogram(x=legit_amounts, nbinsx=60, name="Legitimate",
                        marker_color="steelblue", opacity=0.8),
            row=1, col=1
        )
        fig.add_trace(
            go.Histogram(x=fraud_amounts, nbinsx=60, name="Fraud",
                        marker_color="crimson", opacity=0.8),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Amount (€) - capped at 500€", row=2, col=1)
        fig.update_yaxes(title_text="Count", row=1, col=1)
        fig.update_yaxes(title_text="Count", row=2, col=1)
        fig.update_layout(showlegend=False, height=500)

        st.plotly_chart(fig, use_container_width=True)

    # Class balance
    with col_right:
        st.subheader("Class Balance")

        counts = df["Class"].value_counts().sort_index()
        labels = ["Legitimate", "Fraud"]
        colors = ["steelblue", "crimson"]

        fig = go.Figure(go.Bar(
            x=labels,
            y=counts.values,
            marker_color=colors,
            text=[f"{v:,}" for v in counts.values],
            textposition="outside",
            width=0.4
        ))

        fig.update_layout(
            yaxis_type="log",
            yaxis_title="Count (log scale)",
            title="Log scale - class imbalance",
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)