"""Ablation studies for phenotype prediction validation."""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from config.settings import (
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    LEARNING_RATE,
    MAX_DEPTH,
    NUM_LEAVES,
)


def feature_ablation(X, y, feature_names, center_name="Center"):
    """
    Ablation study: Remove features one by one to assess importance.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Labels.
    feature_names : list
        Names of features.
    center_name : str
        Name of the center for logging.

    Returns
    -------
    results : dict
        Ablation results.
    """
    print(f"\n{'='*60}")
    print(f"  Feature Ablation Study - {center_name}")
    print(f"{'='*60}")

    # Baseline with all features
    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model_full = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        num_leaves=NUM_LEAVES,
        random_state=RANDOM_STATE,
        verbose=-1,
        class_weight="balanced",
    )
    model_full.fit(X_train, y_train)
    y_prob_full = model_full.predict_proba(X_test)
    auroc_full = roc_auc_score(y_test, y_prob_full, multi_class="ovr", average="macro")

    print(f"\n  Baseline AUROC (all features): {auroc_full:.4f}")

    results = {"full": auroc_full}

    # Ablate each feature
    for i, fname in enumerate(feature_names):
        X_reduced = np.delete(X_imputed, i, axis=1)
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_reduced, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )

        model_r = lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            num_leaves=NUM_LEAVES,
            random_state=RANDOM_STATE,
            verbose=-1,
            class_weight="balanced",
        )
        model_r.fit(X_train_r, y_train_r)
        y_prob_r = model_r.predict_proba(X_test_r)
        auroc_r = roc_auc_score(y_test_r, y_prob_r, multi_class="ovr", average="macro")

        drop = auroc_full - auroc_r
        results[fname] = {"auroc": auroc_r, "drop": drop}
        print(f"  Without {fname}: AUROC={auroc_r:.4f} (drop: {drop:.4f})")

    return results


def model_comparison(X, y, center_name="Center"):
    """
    Compare different classification algorithms.

    Parameters
    ----------
    X : np.ndarray
        Full feature matrix.
    y : np.ndarray
        Labels.
    center_name : str
        Name of the center for logging.

    Returns
    -------
    results : dict
        Model comparison results.
    """
    print(f"\n{'='*60}")
    print(f"  Model Comparison Study - {center_name}")
    print(f"{'='*60}")

    imputer = KNNImputer(n_neighbors=5)
    X_imputed = imputer.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    models = {
        "LightGBM": lgb.LGBMClassifier(
            n_estimators=N_ESTIMATORS,
            learning_rate=LEARNING_RATE,
            max_depth=MAX_DEPTH,
            num_leaves=NUM_LEAVES,
            random_state=RANDOM_STATE,
            verbose=-1,
            class_weight="balanced",
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=200,
            max_depth=MAX_DEPTH,
            random_state=RANDOM_STATE,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=RANDOM_STATE,
            class_weight="balanced",
        ),
    }

    results = {}
    for name, model in models.items():
        print(f"\n  Training {name}...")
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)

        try:
            auroc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")
        except ValueError:
            auroc = np.nan

        results[name] = {"auroc": auroc}
        print(f"    AUROC: {auroc:.4f}")

    best_model = max(results.keys(), key=lambda k: results[k]["auroc"])
    print(f"\n  Best model: {best_model}")

    return results


def cross_center_validation(
    X_source, y_source, X_target, y_target, source_name="Source", target_name="Target"
):
    """
    Cross-center validation: train on source, test on target.

    Parameters
    ----------
    X_source, y_source : np.ndarray
        Source center data.
    X_target, y_target : np.ndarray
        Target center data.
    source_name, target_name : str
        Names of centers for logging.

    Returns
    -------
    results : dict
        Cross-validation results.
    """
    print(f"\n{'='*60}")
    print(f"  Cross-Center Validation: {source_name} -> {target_name}")
    print(f"{'='*60}")

    imputer_source = KNNImputer(n_neighbors=5)
    X_source_imputed = imputer_source.fit_transform(X_source)

    imputer_target = KNNImputer(n_neighbors=5)
    X_target_imputed = imputer_target.fit_transform(X_target)

    model = lgb.LGBMClassifier(
        n_estimators=N_ESTIMATORS,
        learning_rate=LEARNING_RATE,
        max_depth=MAX_DEPTH,
        num_leaves=NUM_LEAVES,
        random_state=RANDOM_STATE,
        verbose=-1,
        class_weight="balanced",
    )
    model.fit(X_source_imputed, y_source)

    y_prob = model.predict_proba(X_target_imputed)

    try:
        auroc = roc_auc_score(y_target, y_prob, multi_class="ovr", average="macro")
    except ValueError:
        auroc = np.nan

    print(f"  AUROC: {auroc:.4f}")

    return {"auroc": auroc}
