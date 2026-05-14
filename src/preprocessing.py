import os
import kagglehub
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

# Base directory = project root (one level above src/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_raw_data(local_path=None):
    """
    Load raw credit card dataset.
    """
    if local_path is None:
        local_path = os.path.join(BASE_DIR, "data", "raw", "creditcard.csv")

    if os.path.exists(local_path):
        # Fast path: use local file
        return pd.read_csv(local_path)
    else:
        # Fallback: download via kagglehub
        path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")
        return pd.read_csv(os.path.join(path, "creditcard.csv"))


def scale_features(df):
    """
    Scale 'Amount' and 'Time' features using StandardScaler.
    Returns the scaled DataFrame and the fitted scaler.
    Column names are preserved (Amount, Time).
    """
    scaler = StandardScaler()
    df = df.copy()
    df[["Amount", "Time"]] = scaler.fit_transform(df[["Amount", "Time"]])
    return df, scaler


def split_data(df, test_size=0.2, random_state=42):
    """
    Split dataset into train and test sets.
    Stratified to preserve fraud ratio (~0.17%).
    """
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    return X_train, X_test, y_train, y_test


def apply_smote(X_train, y_train, random_state=42):
    """
    Apply SMOTE oversampling on training set only.
    Balances fraud/legitimate ratio to 1:1 (227451 / 227451).
    Never applied on test set to preserve real-world distribution.
    """
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    return X_resampled, y_resampled


def save_processed_data(X_train, X_test, y_train, y_test,
                        X_resampled, y_resampled,
                        output_dir=None):
    """
    Save all processed datasets to CSV files.
    """
    if output_dir is None:
        output_dir = os.path.join(BASE_DIR, "data", "processed")

    os.makedirs(output_dir, exist_ok=True)

    X_train.to_csv(os.path.join(output_dir, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(output_dir, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(output_dir, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(output_dir, "y_test.csv"), index=False)
    X_resampled.to_csv(os.path.join(output_dir, "X_train_resampled.csv"), index=False)
    y_resampled.to_csv(os.path.join(output_dir, "y_train_resampled.csv"), index=False)

    print(f"Processed data saved to {output_dir}/")


def run_preprocessing():
    """
    Full preprocessing pipeline.
    Run this script directly to generate all processed datasets.
    """
    print("Loading raw data...")
    df = load_raw_data()

    print("Scaling features...")
    df, scaler = scale_features(df)

    print("Splitting data...")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Applying SMOTE...")
    X_resampled, y_resampled = apply_smote(X_train, y_train)

    print("Saving processed data...")
    save_processed_data(X_train, X_test, y_train, y_test, X_resampled, y_resampled)

    print("Done.")
    return scaler


def load_processed_data(data_dir=None):
    """
    Load all processed datasets from CSV files.
    Used by the Streamlit dashboard and model scripts.
    """
    if data_dir is None:
        data_dir = os.path.join(BASE_DIR, "data", "processed")

    X_train = pd.read_csv(os.path.join(data_dir, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(data_dir, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(data_dir, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(data_dir, "y_test.csv")).squeeze()
    X_resampled = pd.read_csv(os.path.join(data_dir, "X_train_resampled.csv"))
    y_resampled = pd.read_csv(os.path.join(data_dir, "y_train_resampled.csv")).squeeze()

    return X_train, X_test, y_train, y_test, X_resampled, y_resampled


if __name__ == "__main__":
    run_preprocessing()