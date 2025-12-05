"""
obesity_risks_ml_tools.py

A collection of phase-appropriate tools for Model Building, Validation, and Prediction
in the 'obesity_risks' competition. Includes both dataset-specific and generic tools.

All tools are pure functions, parameterize all column names, validate inputs,
and return results explicitly. Use only pandas, numpy, scikit-learn, and scipy.

Author: Toolsmith Agent for obesity_risks
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple, Union
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from scipy.stats import mode


############################
# GENERIC MODELING TOOLS   #
############################

def cross_validate_with_multiple_metrics(
    X: pd.DataFrame,
    y: pd.Series,
    estimator,
    cv: int = 5,
    scoring: Optional[Dict[str, Any]] = None,
    random_state: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Performs cross-validation using multiple metrics and returns the mean and std for each.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        estimator: scikit-learn estimator.
        cv (int): Number of folds for cross-validation.
        scoring (dict, optional): Dict of scoring metrics (e.g., {'accuracy': ..., 'f1_macro': ...}).
        random_state (int, optional): Random seed.

    Returns:
        Dict[str, Any]: Dictionary with scores for each metric.

    Raises:
        ValueError: If X or y are empty or have mismatching shapes.

    Return schema:
        {
            'cv_scores': Dict[str, List[float]],
            'mean_scores': Dict[str, float],
            'std_scores': Dict[str, float],
            'n_splits': int,
            'estimator': estimator object,
        }

    Example:
        scores = cross_validate_with_multiple_metrics(X, y, estimator, cv=5, scoring={'accuracy': 'accuracy', 'f1_macro': 'f1_macro'})
    """
    if X.shape[0] == 0 or y.shape[0] == 0:
        raise ValueError("X and y must not be empty.")
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have the same number of rows.")
    if scoring is None:
        scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
    cv_scores = {}
    mean_scores = {}
    std_scores = {}
    for metric_name, metric in scoring.items():
        scores = cross_val_score(
            estimator,
            X,
            y,
            cv=cv,
            scoring=metric,
            n_jobs=None,
        )
        cv_scores[metric_name] = scores.tolist()
        mean_scores[metric_name] = np.mean(scores)
        std_scores[metric_name] = np.std(scores)
    return {
        "cv_scores": cv_scores,
        "mean_scores": mean_scores,
        "std_scores": std_scores,
        "n_splits": cv,
        "estimator": estimator,
    }


def get_classification_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    labels: Optional[List[str]] = None,
    average: str = "macro"
) -> Dict[str, Any]:
    """
    Computes core classification metrics for predictions.

    Args:
        y_true (array-like): Ground truth target values.
        y_pred (array-like): Predicted target values.
        labels (list, optional): List of label names (for consistent order).
        average (str): Averaging method for multi-class metrics.

    Returns:
        Dict[str, Any]: Dictionary with accuracy, f1, confusion matrix, and classification report.

    Raises:
        ValueError: If input lengths mismatch.

    Return schema:
        {
            'accuracy': float,
            'f1_macro': float,
            'confusion_matrix': np.ndarray,
            'classification_report': str,
        }

    Example:
        metrics = get_classification_metrics(y_true, y_pred)
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length.")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average=average, labels=labels)
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    class_report = classification_report(y_true, y_pred, labels=labels)
    return {
        "accuracy": acc,
        "f1_macro": f1,
        "confusion_matrix": conf_mat,
        "classification_report": class_report,
    }


##############################################
# DATASET-SPECIFIC DOMAIN-AWARE MODEL TOOLS  #
##############################################

def obesity_grouped_stratified_kfold(
    X: pd.DataFrame,
    y: pd.Series,
    group_col: Optional[str] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: Optional[int] = None,
) -> StratifiedKFold:
    """
    Returns a StratifiedKFold object. If group_col is provided, warns user and ignores it,
    since group-wise splitting is not supported in this dataset (no group/id dependency).

    Domain rationale:
        Obesity risk is known to be stratified by categories such as BMI_category.
        This function ensures stratified folds for balanced class distribution.

    Args:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target labels.
        group_col (str, optional): Column for group-aware CV (ignored here).
        n_splits (int): Number of CV folds.
        shuffle (bool): Whether to shuffle before splitting.
        random_state (int, optional): Seed.

    Returns:
        StratifiedKFold: StratifiedKFold splitter.

    Raises:
        ValueError: If y is empty or n_splits < 2.

    Return schema:
        StratifiedKFold object (sklearn)

    Example:
        skf = obesity_grouped_stratified_kfold(X, y, n_splits=5)
        for train_idx, val_idx in skf.split(X, y):
            ...
    """
    if y.empty:
        raise ValueError("y must not be empty.")
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2.")
    if group_col is not None:
        import warnings
        warnings.warn("group_col supplied but ignored; no group-dependent splits in this dataset.")
    return StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)


def obesity_prediction_calibrator(
    y_probs: np.ndarray,
    class_labels: List[str],
    method: str = "argmax"
) -> np.ndarray:
    """
    Calibrates predicted probabilities into final class labels.
    Designed for multi-class obesity classification.

    Domain rationale:
        Ensures that calibrated predictions obey the class label schema and
        properly handles ties or uncertain cases in obesity risk prediction.

    Args:
        y_probs (np.ndarray): Array of predicted probabilities (n_samples, n_classes).
        class_labels (List[str]): List of class labels in order of columns in y_probs.
        method (str): 'argmax' (default) or 'top2_random' (breaks ties randomly among top 2).

    Returns:
        np.ndarray: Predicted class labels (shape: n_samples,).

    Raises:
        ValueError: If y_probs shape does not match class_labels.

    Return schema:
        np.ndarray of shape (n_samples,) with class label strings.

    Example:
        preds = obesity_prediction_calibrator(y_probs, class_labels)
    """
    if y_probs.shape[1] != len(class_labels):
        raise ValueError("y_probs columns must match length of class_labels.")
    if method not in {"argmax", "top2_random"}:
        raise ValueError("method must be 'argmax' or 'top2_random'.")
    if method == "argmax":
        preds = np.array([class_labels[i] for i in np.argmax(y_probs, axis=1)])
    else:  # "top2_random"
        preds = []
        for row in y_probs:
            top_val = np.max(row)
            top_idxs = np.where(row == top_val)[0]
            if len(top_idxs) == 1:
                preds.append(class_labels[top_idxs[0]])
            else:
                chosen = np.random.choice(top_idxs)
                preds.append(class_labels[chosen])
        preds = np.array(preds)
    return preds


def obesity_submission_formatter(
    ids: pd.Series,
    preds: Union[np.ndarray, List[str]],
    id_col: str = "id",
    target_col: str = "NObeyesdad"
) -> pd.DataFrame:
    """
    Formats predictions for the obesity_risks competition submission.

    Domain rationale:
        Ensures outputs match required Kaggle schema with id and target columns.
        Handles ordering and formatting reliably for this specific competition.

    Args:
        ids (pd.Series): Series of IDs from the test set.
        preds (np.ndarray or list): Predicted class labels.
        id_col (str): Name for the ID column.
        target_col (str): Name for the prediction column.

    Returns:
        pd.DataFrame: Submission-ready DataFrame.

    Raises:
        ValueError: If lengths do not match.

    Return schema:
        DataFrame with columns [id_col, target_col], shape (n_samples, 2)

    Example:
        sub_df = obesity_submission_formatter(test_df['id'], preds)
    """
    if len(ids) != len(preds):
        raise ValueError("ids and preds must have the same length.")
    df = pd.DataFrame({id_col: ids, target_col: preds})
    return df


##############################################################
# UNIFORM INTERFACE: MODEL WRAPPERS FOR ALGORITHM COMPARISON #
##############################################################

def fit_and_evaluate_logistic_regression(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    *,
    feature_cols: Optional[List[str]] = None,
    target_labels: Optional[List[str]] = None,
    cv: int = 5,
    random_state: Optional[int] = None,
    solver: str = "liblinear",
    max_iter: int = 500,
    return_estimator: bool = True,
) -> Dict[str, Any]:
    """
    Trains and evaluates a Logistic Regression classifier with cross-validation.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training labels.
        X_val (pd.DataFrame, optional): Validation features for out-of-sample evaluation.
        y_val (pd.Series, optional): Validation labels.
        feature_cols (List[str], optional): Subset of features to use. If None, use all.
        target_labels (List[str], optional): List of all class labels (for metrics/report).
        cv (int): Cross-validation folds.
        random_state (int, optional): Seed.
        solver (str): Solver for LogisticRegression.
        max_iter (int): Max iterations for LogisticRegression.
        return_estimator (bool): If True, return fitted estimator.

    Returns:
        Dict[str, Any]: Results including fitted model, CV metrics, and val metrics.

    Raises:
        ValueError: If data is invalid or shapes mismatch.

    Return schema:
        {
            'model_name': str,
            'estimator': LogisticRegression,
            'cv_scores': Dict[str, List[float]],
            'mean_scores': Dict[str, float],
            'std_scores': Dict[str, float],
            'val_metrics': Dict[str, Any] or None,
            'n_features': int,
            'n_samples': int,
        }

    Example:
        results = fit_and_evaluate_logistic_regression(X, y, X_val, y_val, feature_cols=col_list)
    """
    if feature_cols is not None:
        X_use = X[feature_cols]
        Xv_use = X_val[feature_cols] if X_val is not None else None
    else:
        X_use = X.copy()
        Xv_use = X_val.copy() if X_val is not None else None

    lr = LogisticRegression(
        multi_class="auto",
        solver=solver,
        max_iter=max_iter,
        random_state=random_state,
    )
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    cv_results = cross_validate_with_multiple_metrics(X_use, y, lr, cv=cv, scoring=scoring, random_state=random_state)
    lr.fit(X_use, y)
    val_metrics = None
    if Xv_use is not None and y_val is not None:
        y_pred = lr.predict(Xv_use)
        val_metrics = get_classification_metrics(y_val, y_pred, labels=target_labels)
    return {
        "model_name": "LogisticRegression",
        "estimator": lr if return_estimator else None,
        "cv_scores": cv_results["cv_scores"],
        "mean_scores": cv_results["mean_scores"],
        "std_scores": cv_results["std_scores"],
        "val_metrics": val_metrics,
        "n_features": X_use.shape[1],
        "n_samples": X_use.shape[0],
    }


def fit_and_evaluate_random_forest(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    *,
    feature_cols: Optional[List[str]] = None,
    target_labels: Optional[List[str]] = None,
    cv: int = 5,
    n_estimators: int = 100,
    max_depth: Optional[int] = None,
    random_state: Optional[int] = None,
    return_estimator: bool = True,
) -> Dict[str, Any]:
    """
    Trains and evaluates a Random Forest classifier with cross-validation.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training labels.
        X_val (pd.DataFrame, optional): Validation features.
        y_val (pd.Series, optional): Validation labels.
        feature_cols (List[str], optional): Subset of features to use. If None, use all.
        target_labels (List[str], optional): List of class labels (for metrics).
        cv (int): Number of CV folds.
        n_estimators (int): Number of trees.
        max_depth (int, optional): Max tree depth.
        random_state (int, optional): Seed.
        return_estimator (bool): If True, return fitted estimator.

    Returns:
        Dict[str, Any]: Results including fitted model, CV metrics, and val metrics.

    Raises:
        ValueError: If data is invalid.

    Return schema:
        {
            'model_name': str,
            'estimator': RandomForestClassifier,
            'cv_scores': Dict[str, List[float]],
            'mean_scores': Dict[str, float],
            'std_scores': Dict[str, float],
            'val_metrics': Dict[str, Any] or None,
            'n_features': int,
            'n_samples': int,
        }

    Example:
        results = fit_and_evaluate_random_forest(X, y, X_val, y_val, feature_cols=col_list)
    """
    if feature_cols is not None:
        X_use = X[feature_cols]
        Xv_use = X_val[feature_cols] if X_val is not None else None
    else:
        X_use = X.copy()
        Xv_use = X_val.copy() if X_val is not None else None

    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    cv_results = cross_validate_with_multiple_metrics(X_use, y, rf, cv=cv, scoring=scoring, random_state=random_state)
    rf.fit(X_use, y)
    val_metrics = None
    if Xv_use is not None and y_val is not None:
        y_pred = rf.predict(Xv_use)
        val_metrics = get_classification_metrics(y_val, y_pred, labels=target_labels)
    return {
        "model_name": "RandomForestClassifier",
        "estimator": rf if return_estimator else None,
        "cv_scores": cv_results["cv_scores"],
        "mean_scores": cv_results["mean_scores"],
        "std_scores": cv_results["std_scores"],
        "val_metrics": val_metrics,
        "n_features": X_use.shape[1],
        "n_samples": X_use.shape[0],
    }


def fit_and_evaluate_svc(
    X: pd.DataFrame,
    y: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    *,
    feature_cols: Optional[List[str]] = None,
    target_labels: Optional[List[str]] = None,
    cv: int = 5,
    kernel: str = "rbf",
    C: float = 1.0,
    gamma: Union[str, float] = "scale",
    random_state: Optional[int] = None,
    return_estimator: bool = True,
) -> Dict[str, Any]:
    """
    Trains and evaluates a Support Vector Classifier (SVC) with cross-validation.

    Args:
        X (pd.DataFrame): Training features.
        y (pd.Series): Training labels.
        X_val (pd.DataFrame, optional): Validation features.
        y_val (pd.Series, optional): Validation labels.
        feature_cols (List[str], optional): Subset of features to use. If None, use all.
        target_labels (List[str], optional): List of class labels (for metrics).
        cv (int): Number of CV folds.
        kernel (str): SVC kernel type.
        C (float): Regularization parameter.
        gamma (str or float): Kernel coefficient.
        random_state (int, optional): Seed (for shuffling).
        return_estimator (bool): If True, return fitted estimator.

    Returns:
        Dict[str, Any]: Results including fitted model, CV metrics, and val metrics.

    Raises:
        ValueError: If data is invalid.

    Return schema:
        {
            'model_name': str,
            'estimator': SVC,
            'cv_scores': Dict[str, List[float]],
            'mean_scores': Dict[str, float],
            'std_scores': Dict[str, float],
            'val_metrics': Dict[str, Any] or None,
            'n_features': int,
            'n_samples': int,
        }

    Example:
        results = fit_and_evaluate_svc(X, y, X_val, y_val, feature_cols=col_list)
    """
    if feature_cols is not None:
        X_use = X[feature_cols]
        Xv_use = X_val[feature_cols] if X_val is not None else None
    else:
        X_use = X.copy()
        Xv_use = X_val.copy() if X_val is not None else None

    svc = SVC(
        kernel=kernel,
        C=C,
        gamma=gamma,
        probability=True,  # For probability outputs if needed
        random_state=random_state,
    )
    scoring = {"accuracy": "accuracy", "f1_macro": "f1_macro"}
    cv_results = cross_validate_with_multiple_metrics(X_use, y, svc, cv=cv, scoring=scoring, random_state=random_state)
    svc.fit(X_use, y)
    val_metrics = None
    if Xv_use is not None and y_val is not None:
        y_pred = svc.predict(Xv_use)
        val_metrics = get_classification_metrics(y_val, y_pred, labels=target_labels)
    return {
        "model_name": "SVC",
        "estimator": svc if return_estimator else None,
        "cv_scores": cv_results["cv_scores"],
        "mean_scores": cv_results["mean_scores"],
        "std_scores": cv_results["std_scores"],
        "val_metrics": val_metrics,
        "n_features": X_use.shape[1],
        "n_samples": X_use.shape[0],
    }


####################################
# UTILITY: ENSEMBLING AND WRAPPING #
####################################

def soft_voting_ensemble_predict(
    estimators: List[Any],
    X: pd.DataFrame,
    class_labels: List[str],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """
    Makes ensemble predictions using soft voting (average predicted probabilities).

    Args:
        estimators (List[sklearn estimator]): List of fitted estimators with predict_proba().
        X (pd.DataFrame): Data to predict on.
        class_labels (List[str]): List of class labels (in order).
        weights (List[float], optional): List of ensemble weights. If None, use equal weights.

    Returns:
        np.ndarray: Array of predicted class labels.

    Raises:
        ValueError: If any estimator lacks predict_proba, or weights mismatch.

    Return schema:
        np.ndarray of shape (n_samples,) with predicted class labels.

    Example:
        preds = soft_voting_ensemble_predict([lr, rf, svc], X_test, class_labels)
    """
    n_estimators = len(estimators)
    if n_estimators == 0:
        raise ValueError("Must provide at least one estimator.")
    if weights is None:
        weights = [1.0] * n_estimators
    if len(weights) != n_estimators:
        raise ValueError("Length of weights must match number of estimators.")
    probas = None
    for est, w in zip(estimators, weights):
        if not hasattr(est, "predict_proba"):
            raise ValueError("All estimators must implement predict_proba.")
        p = est.predict_proba(X)
        if probas is None:
            probas = np.zeros_like(p, dtype=float)
        probas += w * p
    probas /= np.sum(weights)
    preds = obesity_prediction_calibrator(probas, class_labels, method="argmax")
    return preds


#####################
# END OF MODULE     #
#####################