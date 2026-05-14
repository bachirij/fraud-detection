import os
import numpy as np
import pandas as pd
import joblib

# Base directory = project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def save_model(model, model_name, output_dir=None):
    """
    Save a trained model to disk using joblib.
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{model_name}.joblib")
    joblib.dump(model, path)
    print(f"Model saved to {path}")


def load_model(model_name, models_dir=None):
    """
    Load a trained model from disk.
    """
    if models_dir is None:
        models_dir = os.path.join(BASE_DIR, "models")
    path = os.path.join(models_dir, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No model found at {path}. Run train_all_models() first.")
    return joblib.load(path)


def predict_transaction(transaction, model, scaler):
    """
    Predict whether a single transaction is fraudulent.
    transaction: dict with keys Time, V1-V28, Amount.
    Returns prediction (0/1) and fraud probability.
    """
    df = pd.DataFrame([transaction])
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    prediction = model.predict(df)[0]
    fraud_probability = model.predict_proba(df)[0][1]

    return {
        "prediction": int(prediction),
        "label": "Fraud" if prediction == 1 else "Legitimate",
        "fraud_probability": round(float(fraud_probability), 4)
    }


def predict_batch(df, model, scaler):
    """
    Predict fraud on a batch of transactions.
    df: DataFrame with columns Time, V1-V28, Amount.
    Returns the original DataFrame with added columns: prediction, label, fraud_probability.
    """
    df = df.copy()
    df[["Amount", "Time"]] = scaler.transform(df[["Amount", "Time"]])

    df["prediction"] = model.predict(df)
    df["fraud_probability"] = model.predict_proba(df)[:, 1].round(4)
    df["label"] = df["prediction"].map({0: "Legitimate", 1: "Fraud"})

    return df


if __name__ == "__main__":
    from preprocessing import load_processed_data, scale_features, load_raw_data
    from models import train_all_models

    # Train and save all models
    X_train, X_test, y_train, y_test, X_resampled, y_resampled = load_processed_data()
    models = train_all_models(X_train, y_train, X_resampled, y_resampled)

    for model_name, model in models.items():
        save_model(model, model_name)

    # Save scaler
    df_raw = load_raw_data()
    _, scaler = scale_features(df_raw)
    save_model(scaler, "scaler")

    # Test predict_transaction() with a real row from X_test
    xgb = load_model("xgboost")
    scaler = load_model("scaler")

    sample = X_test.iloc[0].to_dict()
    # Reverse scaling to simulate a raw incoming transaction
    amount_scaled, time_scaled = sample["Amount"], sample["Time"]
    sample["Amount"] = amount_scaled * scaler.scale_[1] + scaler.mean_[1]
    sample["Time"] = time_scaled * scaler.scale_[0] + scaler.mean_[0]

    result = predict_transaction(sample, xgb, scaler)
    print(f"\nSample transaction prediction:")
    print(f"  Label            : {result['label']}")
    print(f"  Fraud probability: {result['fraud_probability']}")
    print(f"  Actual label     : {'Fraud' if y_test.iloc[0] == 1 else 'Legitimate'}")