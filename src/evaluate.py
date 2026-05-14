import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)


def evaluate_model(model, X_test, y_test, model_name="Model", save_path=None):
    """
    Evaluate a single model: classification report, confusion matrix, ROC curve.
    If save_path is provided, saves the figure instead of displaying it.
    """
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Classification report
    print(f"\n{'='*50}")
    print(f"Model: {model_name}")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"]))

    # AUC-ROC
    auc = roc_auc_score(y_test, y_proba)
    print(f"AUC-ROC: {auc:.4f}")

    # Confusion matrix + ROC curve side by side
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(model_name, fontsize=14, fontweight="bold")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=["Legitimate", "Fraud"],
                yticklabels=["Legitimate", "Fraud"])
    axes[0].set_title("Confusion Matrix")
    axes[0].set_ylabel("Actual")
    axes[0].set_xlabel("Predicted")

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f"AUC = {auc:.4f}")
    axes[1].plot([0, 1], [0, 1], "k--", label="Random")
    axes[1].set_title("ROC Curve")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return {"model": model_name, "auc_roc": auc, "y_pred": y_pred, "y_proba": y_proba}


def compare_models(models, X_test, y_test):
    """
    Evaluate all models and return a comparative DataFrame.
    Columns: Model, Precision, Recall, F1, AUC-ROC, FN, FP
    """
    from sklearn.metrics import precision_score, recall_score, f1_score

    rows = []
    for model_name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        rows.append({
            "Model": model_name,
            "Precision": round(precision_score(y_test, y_pred), 2),
            "Recall": round(recall_score(y_test, y_pred), 2),
            "F1": round(f1_score(y_test, y_pred), 2),
            "AUC-ROC": round(roc_auc_score(y_test, y_proba), 4),
            "FN": fn,
            "FP": fp
        })

    df = pd.DataFrame(rows).sort_values("F1", ascending=False).reset_index(drop=True)
    return df


def plot_roc_curves(models, X_test, y_test, save_path=None):
    """
    Plot superimposed ROC curves for all models.
    If save_path is provided, saves the figure instead of displaying it.
    """
    plt.figure(figsize=(8, 6))

    for model_name, model in models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        auc = roc_auc_score(y_test, y_proba)
        plt.plot(fpr, tpr, label=f"{model_name} (AUC = {auc:.4f})")

    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.title("ROC Curves - All Models")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend(loc="lower right")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


if __name__ == "__main__":
    from preprocessing import load_processed_data
    from models import train_all_models

    X_train, X_test, y_train, y_test, X_resampled, y_resampled = load_processed_data()
    models = train_all_models(X_train, y_train, X_resampled, y_resampled)

    for model_name, model in models.items():
        evaluate_model(model, X_test, y_test, model_name=model_name)

    print("\nComparative results:")
    print(compare_models(models, X_test, y_test))

    plot_roc_curves(models, X_test, y_test)