from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier


def train_logistic_regression(X_train_resampled, y_resampled):
    """
    Train Logistic Regression on SMOTE-resampled data.
    max_iter=1000 to ensure convergence on this dataset.
    """
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_resampled, y_resampled)
    return model


def train_decision_tree(X_train_resampled, y_resampled):
    """
    Train Decision Tree on SMOTE-resampled data.
    """
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_resampled, y_resampled)
    return model


def train_random_forest(X_train_resampled, y_resampled):
    """
    Train Random Forest on SMOTE-resampled data.
    n_jobs=-1 to use all available CPU cores.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train_resampled, y_resampled)
    return model


def train_xgboost(X_train, y_train):
    """
    Train XGBoost on original (non-resampled) training data.
    scale_pos_weight handles class imbalance natively (227451 / 394 ≈ 577).
    """
    model = XGBClassifier(
        scale_pos_weight=577,
        random_state=42,
        eval_metric="logloss",
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    return model


def train_all_models(X_train, y_train, X_resampled, y_resampled):
    """
    Train all models and return them as a dictionary.
    LR, DT, RF use SMOTE-resampled data.
    XGBoost uses original training data with scale_pos_weight.
    """
    print("Training Logistic Regression...")
    lr = train_logistic_regression(X_resampled, y_resampled)

    print("Training Decision Tree...")
    dt = train_decision_tree(X_resampled, y_resampled)

    print("Training Random Forest...")
    rf = train_random_forest(X_resampled, y_resampled)

    print("Training XGBoost...")
    xgb = train_xgboost(X_train, y_train)

    print("All models trained.")
    return {
        "logistic_regression": lr,
        "decision_tree": dt,
        "random_forest": rf,
        "xgboost": xgb
    }


if __name__ == "__main__":
    from preprocessing import load_processed_data

    X_train, X_test, y_train, y_test, X_resampled, y_resampled = load_processed_data()
    models = train_all_models(X_train, y_train, X_resampled, y_resampled)
    