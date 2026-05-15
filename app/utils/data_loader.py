import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "..", "src"))

import streamlit as st
from preprocessing import load_processed_data
from predict import load_model

@st.cache_data
def get_data():
    """
    Load processed train/test splits from data/processed/
    """
    X_train, X_test, y_train, y_test, X_resampled, y_resampled = load_processed_data()
    return X_train, X_test, y_train, y_test, X_resampled, y_resampled

@st.cache_resource
def get_models():
    """
    Load all trained models from models/
    """
    models = {
        "Logistic Regression": load_model("logistic_regression"),
        "Decision Tree":       load_model("decision_tree"),
        "Random Forest":       load_model("random_forest"),
        "XGBoost":             load_model("xgboost"),
    }
    return models

@st.cache_resource
def get_scaler():
    """
    Load the fitted StandardScaler
    """
    return load_model("scaler")