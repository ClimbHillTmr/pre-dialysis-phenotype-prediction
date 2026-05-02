"""Model training and evaluation for phenotype prediction."""

import numpy as np
import lightgbm as lgb
import shap
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import os

from config.settings import (
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    LEARNING_RATE,
    MAX_DEPTH,
    NUM_LEAVES,
    FIGURE_DPI,
)


def train_lightgbm(X, y, center_name="Center", use_early_stopping=True):
    """
    Train LightGBM classifier with class weight balancing.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    center_name : str
        Name of the center for logging.
    use_early_stopping : bool
        Whether to use early stopping to prevent overfitting.

    Returns
    -------
    model : lgb.LGBMClassifier
        Trained model.
    X_train, X_test, y_train, y_test : np.ndarray
        Train/test splits.
    y_prob : np.ndarray
        Predicted probabilities.
    """
    print(f"\n{'='*60}")
    print(f"  {center_name} Model Training")
    print(f"{'='*60}")

    # Train/test split FIRST to prevent data leakage
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # KNN imputation AFTER split to prevent data leakage
    imputer = KNNImputer(n_neighbors=5)
    X_train_imputed = imputer.fit_transform(X_train)
    X_test_imputed = imputer.transform(X_test)

    # Feature scaling for cross-center consistency
    scaler = RobustScaler()
    X_train_scaled = scaler.fit_transform(X_train_imputed)
    X_test_scaled = scaler.transform(X_test_imputed)

    print(f"  Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"  Class distribution: {np.bincount(y_train)}")

    # Train LightGBM with early stopping
    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        num_leaves=NUM_LEAVES,
        random_state=RANDOM_STATE,
        verbose=-1,
        class_weight="balanced",
    )

    if use_early_stopping:
        callbacks = [lgb.early_stopping(stopping_rounds=50)]
    else:
        callbacks = []

    model.fit(
        X_train_scaled, y_train, eval_set=[(X_test_scaled, y_test)], callbacks=callbacks
    )

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n  Test Accuracy: {accuracy:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred))

    return model, X_train, X_test, y_train, y_test, y_prob


def compute_metrics(y_true, y_prob, y_pred, n_classes=4):
    """Compute comprehensive evaluation metrics."""
    metrics = {}

    # AUROC (macro average)
    try:
        auroc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
        metrics["auroc_macro"] = auroc
    except ValueError:
        metrics["auroc_macro"] = np.nan

    # Per-class AUROC
    auroc_per_class = []
    for i in range(n_classes):
        try:
            y_binary = (y_true == i).astype(int)
            auroc_i = roc_auc_score(y_binary, y_prob[:, i])
            auroc_per_class.append(auroc_i)
        except ValueError:
            auroc_per_class.append(np.nan)
    metrics["auroc_per_class"] = auroc_per_class

    # AUPRC (macro average)
    try:
        auprc = average_precision_score(y_true, y_prob, average="macro")
        metrics["auprc_macro"] = auprc
    except ValueError:
        metrics["auprc_macro"] = np.nan

    # Per-class AUPRC
    auprc_per_class = []
    for i in range(n_classes):
        try:
            y_binary = (y_true == i).astype(int)
            auprc_i = average_precision_score(y_binary, y_prob[:, i])
            auprc_per_class.append(auprc_i)
        except ValueError:
            auprc_per_class.append(np.nan)
    metrics["auprc_per_class"] = auprc_per_class

    # Accuracy
    metrics["accuracy"] = accuracy_score(y_true, y_pred)

    # Confusion matrix
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred)

    return metrics


def plot_feature_importance(model, feature_names, output_path, top_n=10):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[-top_n:]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(indices)), importances[indices], color="#2980b9")
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title("Top Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_shap_summary(model, X_test, feature_names, output_path):
    """Generate SHAP summary plot."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # For multi-class, use the mean absolute SHAP values
    if isinstance(shap_values, list):
        shap_values = np.mean([np.abs(sv) for sv in shap_values], axis=0)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(
        shap_values, X_test, feature_names=feature_names, show=False, plot_size=None
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")


def plot_calibration_curve(y_true, y_prob, n_classes=4, output_path="calibration.pdf"):
    """Plot calibration curves for each class."""
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")

    for i in range(n_classes):
        y_binary = (y_true == i).astype(int)
        prob_true, prob_pred = calibration_curve(y_binary, y_prob[:, i], n_bins=10)
        ax.plot(prob_pred, prob_true, "o-", label=f"Class {i}")

    ax.set_xlabel("Mean predicted probability", fontsize=12)
    ax.set_ylabel("Fraction of positives", fontsize=12)
    ax.set_title("Calibration Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=FIGURE_DPI, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {output_path}")
