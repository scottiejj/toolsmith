## cross_validate_with_multiple_metrics

**Name:** cross_validate_with_multiple_metrics  
**Description:** Performs cross-validation using multiple metrics and returns the mean and standard deviation for each metric.  
**Applicable Situations:** Comparing model performance across several metrics using cross-validation.

**Parameters:**
- `X`:
  - **Type:** `pd.DataFrame`
  - **Description:** Feature matrix for model input.
- `y`:
  - **Type:** `pd.Series`
  - **Description:** Target vector.
- `estimator`:
  - **Type:** estimator object
  - **Description:** scikit-learn estimator to fit and evaluate.
- `cv`:
  - **Type:** `int`
  - **Description:** Number of folds for cross-validation.
  - **Default:** `5`
- `scoring`:
  - **Type:** `dict` | `None`
  - **Description:** Dictionary of scoring metrics, e.g., `{'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}`.
  - **Default:** `None`
- `random_state`:
  - **Type:** `int` | `None`
  - **Description:** Random seed for reproducibility.
  - **Default:** `None`

**Required:** `X`, `y`, `estimator`  
**Result:** Dictionary containing cross-validation scores, means, and standard deviations for each metric  
**Notes:**
- Raises `ValueError` if X or y are empty or if their shapes mismatch.
- The estimator is not fitted after running this function.
- Metrics dictionary keys are the same as provided in the `scoring` argument.
---
## get_classification_metrics

**Name:** get_classification_metrics  
**Description:** Computes core classification metrics (accuracy, macro F1, confusion matrix, and classification report) for predictions.  
**Applicable Situations:** Evaluating predictions on multiclass classification tasks.

**Parameters:**
- `y_true`:
  - **Type:** `np.ndarray` | `pd.Series`
  - **Description:** Ground truth target values.
- `y_pred`:
  - **Type:** `np.ndarray` | `pd.Series`
  - **Description:** Predicted target values.
- `labels`:
  - **Type:** `list` | `None`
  - **Description:** List of label names (optional, for consistent order).
  - **Default:** `None`
- `average`:
  - **Type:** `str`
  - **Description:** Averaging method for multi-class metrics.
  - **Default:** `macro`

**Required:** `y_true`, `y_pred`  
**Result:** Dictionary containing accuracy, F1 macro, confusion matrix, and classification report  
**Notes:**
- Raises `ValueError` if input lengths mismatch.
- The function uses the provided labels for metrics if given.
---
## obesity_grouped_stratified_kfold

**Name:** obesity_grouped_stratified_kfold  
**Description:** Returns a `StratifiedKFold` object, ensuring stratified folds for balanced class distribution among obesity risk levels. Ignores group_col and warns if provided, since group dependency is not present in this dataset.  
**Applicable Situations:** Cross-validation in multiclass obesity classification with balanced class splits.

**Parameters:**
- `X`:
  - **Type:** `pd.DataFrame`
  - **Description:** Feature matrix.
- `y`:
  - **Type:** `pd.Series`
  - **Description:** Target labels.
- `group_col`:
  - **Type:** `str` | `None`
  - **Description:** Optional column for group-aware cross-validation (ignored).
  - **Default:** `None`
- `n_splits`:
  - **Type:** `int`
  - **Description:** Number of cross-validation folds.
  - **Default:** `5`
- `shuffle`:
  - **Type:** `bool`
  - **Description:** Whether to shuffle before splitting.
  - **Default:** `True`
- `random_state`:
  - **Type:** `int` | `None`
  - **Description:** Random seed.
  - **Default:** `None`

**Required:** `X`, `y`  
**Result:** StratifiedKFold splitter object  
**Notes:**
- Raises `ValueError` if y is empty or n_splits < 2.
- Warns if group_col is supplied; no group-dependent splits for this dataset.
- Domain rationale: Obesity risk classification requires balanced class splits due to class imbalance risks; stratification by target is especially important in this competition.
---
## obesity_prediction_calibrator

**Name:** obesity_prediction_calibrator  
**Description:** Calibrates predicted class probabilities into final class labels for multi-class obesity classification, with optional tie-breaking methods.  
**Applicable Situations:** Converting predicted probabilities (e.g., from softmax or predict_proba) to discrete class label predictions.

**Parameters:**
- `y_probs`:
  - **Type:** `np.ndarray`
  - **Description:** Array of predicted probabilities (shape: n_samples, n_classes).
- `class_labels`:
  - **Type:** `list`
  - **Description:** List of class labels in the order of y_probs columns.
- `method`:
  - **Type:** `str`
  - **Description:** Method to convert probabilities to labels.
  - **Enum:** `argmax` | `top2_random`
  - **Default:** `argmax`

**Required:** `y_probs`, `class_labels`  
**Result:** Numpy array of predicted class labels  
**Notes:**
- Raises `ValueError` if y_probs columns do not match number of class_labels, or if method is invalid.
- 'argmax' assigns the class with the highest probability; 'top2_random' randomly selects among tied top probabilities.
- Domain rationale: Obesity risk classes require precise mapping from probability to label, and this tool ensures competition-compliant prediction logic.
---
## obesity_submission_formatter

**Name:** obesity_submission_formatter  
**Description:** Formats predictions and IDs into the required Kaggle submission schema for the obesity_risks competition.  
**Applicable Situations:** Preparing prediction outputs for submission, ensuring correct column order and naming.

**Parameters:**
- `ids`:
  - **Type:** `pd.Series`
  - **Description:** Series of IDs from the test set.
- `preds`:
  - **Type:** `np.ndarray` | `list`
  - **Description:** Predicted class labels.
- `id_col`:
  - **Type:** `str`
  - **Description:** Name for the ID column in the output.
  - **Default:** `id`
- `target_col`:
  - **Type:** `str`
  - **Description:** Name for the prediction/target column in the output.
  - **Default:** `NObeyesdad`

**Required:** `ids`, `preds`  
**Result:** Submission-ready DataFrame with two columns  
**Notes:**
- Raises `ValueError` if lengths of ids and preds do not match.
- Domain rationale: Ensures output format is fully compliant with the competition's submission requirements, reducing risk of format-based disqualification.
---
## fit_and_evaluate_logistic_regression

**Name:** fit_and_evaluate_logistic_regression  
**Description:** Trains and evaluates a Logistic Regression classifier with cross-validation, and (optionally) evaluates on a provided validation set. Returns standardized results for downstream comparison.  
**Applicable Situations:** Comparing or benchmarking logistic regression models for multiclass classification.

**Parameters:**
- `X`:
  - **Type:** `pd.DataFrame`
  - **Description:** Training features.
- `y`:
  - **Type:** `pd.Series`
  - **Description:** Training labels.
- `X_val`:
  - **Type:** `pd.DataFrame` | `None`
  - **Description:** Validation features for out-of-sample evaluation.
  - **Default:** `None`
- `y_val`:
  - **Type:** `pd.Series` | `None`
  - **Description:** Validation labels.
  - **Default:** `None`
- `feature_cols`:
  - **Type:** `list` | `None`
  - **Description:** Subset of features to use; if None, use all.
  - **Default:** `None`
- `target_labels`:
  - **Type:** `list` | `None`
  - **Description:** List of all class labels for metrics/report.
  - **Default:** `None`
- `cv`:
  - **Type:** `int`
  - **Description:** Cross-validation folds.
  - **Default:** `5`
- `random_state`:
  - **Type:** `int` | `None`
  - **Description:** Seed.
  - **Default:** `None`
- `solver`:
  - **Type:** `str`
  - **Description:** Solver for LogisticRegression.
  - **Default:** `liblinear`
- `max_iter`:
  - **Type:** `int`
  - **Description:** Maximum iterations for LogisticRegression.
  - **Default:** `500`
- `return_estimator`:
  - **Type:** `bool`
  - **Description:** Whether to return the fitted estimator.
  - **Default:** `True`

**Required:** `X`, `y`  
**Result:** Dictionary containing fitted estimator, CV metrics, optional validation metrics, and data shape info  
**Notes:**
- Raises `ValueError` if data is invalid or shapes mismatch.
- Supports multiclass and binary targets.
- Returned dictionary keys: model_name, estimator, cv_scores, mean_scores, std_scores, val_metrics, n_features, n_samples.
---
## fit_and_evaluate_random_forest

**Name:** fit_and_evaluate_random_forest  
**Description:** Trains and evaluates a Random Forest classifier with cross-validation, and (optionally) evaluates on a provided validation set. Returns standardized results for downstream comparison.  
**Applicable Situations:** Comparing or benchmarking random forest models for multiclass classification.

**Parameters:**
- `X`:
  - **Type:** `pd.DataFrame`
  - **Description:** Training features.
- `y`:
  - **Type:** `pd.Series`
  - **Description:** Training labels.
- `X_val`:
  - **Type:** `pd.DataFrame` | `None`
  - **Description:** Validation features for out-of-sample evaluation.
  - **Default:** `None`
- `y_val`:
  - **Type:** `pd.Series` | `None`
  - **Description:** Validation labels.
  - **Default:** `None`
- `feature_cols`:
  - **Type:** `list` | `None`
  - **Description:** Subset of features to use; if None, use all.
  - **Default:** `None`
- `target_labels`:
  - **Type:** `list` | `None`
  - **Description:** List of all class labels for metrics/report.
  - **Default:** `None`
- `cv`:
  - **Type:** `int`
  - **Description:** Cross-validation folds.
  - **Default:** `5`
- `n_estimators`:
  - **Type:** `int`
  - **Description:** Number of trees.
  - **Default:** `100`
- `max_depth`:
  - **Type:** `int` | `None`
  - **Description:** Maximum tree depth.
  - **Default:** `None`
- `random_state`:
  - **Type:** `int` | `None`
  - **Description:** Seed.
  - **Default:** `None`
- `return_estimator`:
  - **Type:** `bool`
  - **Description:** Whether to return the fitted estimator.
  - **Default:** `True`

**Required:** `X`, `y`  
**Result:** Dictionary containing fitted estimator, CV metrics, optional validation metrics, and data shape info  
**Notes:**
- Raises `ValueError` if data is invalid or shapes mismatch.
- Returned dictionary keys: model_name, estimator, cv_scores, mean_scores, std_scores, val_metrics, n_features, n_samples.
---
## fit_and_evaluate_svc

**Name:** fit_and_evaluate_svc  
**Description:** Trains and evaluates a Support Vector Classifier (SVC) with cross-validation, and (optionally) evaluates on a provided validation set. Returns standardized results for downstream comparison.  
**Applicable Situations:** Comparing or benchmarking SVC models for multiclass classification.

**Parameters:**
- `X`:
  - **Type:** `pd.DataFrame`
  - **Description:** Training features.
- `y`:
  - **Type:** `pd.Series`
  - **Description:** Training labels.
- `X_val`:
  - **Type:** `pd.DataFrame` | `None`
  - **Description:** Validation features for out-of-sample evaluation.
  - **Default:** `None`
- `y_val`:
  - **Type:** `pd.Series` | `None`
  - **Description:** Validation labels.
  - **Default:** `None`
- `feature_cols`:
  - **Type:** `list` | `None`
  - **Description:** Subset of features to use; if None, use all.
  - **Default:** `None`
- `target_labels`:
  - **Type:** `list` | `None`
  - **Description:** List of all class labels for metrics/report.
  - **Default:** `None`
- `cv`:
  - **Type:** `int`
  - **Description:** Cross-validation folds.
  - **Default:** `5`
- `kernel`:
  - **Type:** `str`
  - **Description:** SVC kernel type.
  - **Default:** `rbf`
- `C`:
  - **Type:** `float`
  - **Description:** Regularization parameter.
  - **Default:** `1.0`
- `gamma`:
  - **Type:** `str` | `float`
  - **Description:** Kernel coefficient.
  - **Default:** `scale`
- `random_state`:
  - **Type:** `int` | `None`
  - **Description:** Seed (used for shuffling).
  - **Default:** `None`
- `return_estimator`:
  - **Type:** `bool`
  - **Description:** Whether to return the fitted estimator.
  - **Default:** `True`

**Required:** `X`, `y`  
**Result:** Dictionary containing fitted estimator, CV metrics, optional validation metrics, and data shape info  
**Notes:**
- Raises `ValueError` if data is invalid or shapes mismatch.
- Returned dictionary keys: model_name, estimator, cv_scores, mean_scores, std_scores, val_metrics, n_features, n_samples.
---
## soft_voting_ensemble_predict

**Name:** soft_voting_ensemble_predict  
**Description:** Makes ensemble predictions by averaging predicted probabilities (soft voting) from multiple fitted estimators, then mapping to class labels.  
**Applicable Situations:** Ensembling multiple probabilistic classifiers for multiclass predictions.

**Parameters:**
- `estimators`:
  - **Type:** `list`
  - **Description:** List of fitted estimators implementing `predict_proba`.
- `X`:
  - **Type:** `pd.DataFrame`
  - **Description:** Data to predict on.
- `class_labels`:
  - **Type:** `list`
  - **Description:** List of class labels, in order corresponding to predicted probabilities.
- `weights`:
  - **Type:** `list` | `None`
  - **Description:** Ensemble weights for each estimator; if None, use equal weights.
  - **Default:** `None`

**Required:** `estimators`, `X`, `class_labels`  
**Result:** Numpy array of predicted class labels  
**Notes:**
- Raises `ValueError` if any estimator lacks predict_proba, or if the number of weights does not match the number of estimators.
- Uses obesity_prediction_calibrator to map ensemble probabilities to class labels.
---