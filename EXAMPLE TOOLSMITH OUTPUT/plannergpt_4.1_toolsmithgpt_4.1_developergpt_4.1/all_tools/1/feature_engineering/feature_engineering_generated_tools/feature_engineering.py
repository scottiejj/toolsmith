import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Any, Union
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from scipy.stats import zscore

####################################
# Feature Engineering Tools Module #
# For: obesity_risks Competition   #
# Phase: Feature Engineering       #
####################################

# ===============================
# ===== DATASET-SPECIFIC TOOLS ===
# ===============================

def create_bmi_category_feature(
    df: pd.DataFrame,
    bmi_col: str = "BMI",
    new_col: str = "BMI_category"
) -> pd.DataFrame:
    """
    Create a BMI category feature based on WHO BMI thresholds.

    Args:
        df (pd.DataFrame): Input DataFrame.
        bmi_col (str): Column name for BMI.
        new_col (str): Name for the new BMI category column.

    Returns:
        pd.DataFrame: DataFrame with new BMI category column.

    Raises:
        ValueError: If bmi_col not in df.

    Notes:
        - Dataset-specific tool.
        - Domain rationale: Encapsulates medical BMI thresholds, making it easier to combine with other features or model non-linear relationships.
        - BMI categories:
            - Underweight: < 18.5
            - Normal weight: 18.5 - 24.9
            - Overweight: 25 - 29.9
            - Obesity I: 30 - 34.9
            - Obesity II: 35 - 39.9
            - Obesity III: >= 40

    Example:
        df = create_bmi_category_feature(df, bmi_col="BMI", new_col="BMI_category")
    """
    if bmi_col not in df.columns:
        raise ValueError(f"Column '{bmi_col}' not found in DataFrame.")
    bins = [-np.inf, 18.5, 25, 30, 35, 40, np.inf]
    labels = [
        "Underweight", "Normal_weight", "Overweight",
        "Obesity_I", "Obesity_II", "Obesity_III"
    ]
    df = df.copy()
    df[new_col] = pd.cut(df[bmi_col], bins=bins, labels=labels, right=False)
    return df

def create_lifestyle_risk_score(
    df: pd.DataFrame,
    favc_col: str = "FAVC",
    caec_col: str = "CAEC",
    calc_col: str = "CALC",
    scc_col: str = "SCC",
    smoke_col: str = "SMOKE",
    fcvc_col: str = "FCVC",
    ch2o_col: str = "CH2O",
    faf_col: str = "FAF",
    tue_col: str = "TUE",
    new_col: str = "lifestyle_risk_score"
) -> pd.DataFrame:
    """
    Create a composite lifestyle risk score based on dietary and behavioral factors.

    Args:
        df (pd.DataFrame): Input DataFrame.
        favc_col (str): Column for high-caloric food consumption (yes/no).
        caec_col (str): Column for food between meals.
        calc_col (str): Column for alcohol consumption.
        scc_col (str): Column for caloric monitoring (yes/no).
        smoke_col (str): Column for smoking (yes/no).
        fcvc_col (str): Frequency of vegetable consumption (higher is better).
        ch2o_col (str): Water intake (higher is better).
        faf_col (str): Physical activity frequency (higher is better).
        tue_col (str): Time using technology (higher is worse).
        new_col (str): Name for the new lifestyle risk score column.

    Returns:
        pd.DataFrame: DataFrame with the new lifestyle risk score column.

    Raises:
        ValueError: If any required column is missing.

    Notes:
        - Dataset-specific tool.
        - Domain rationale: Obesity is influenced by combined lifestyle factors; this score summarizes risk-promoting behaviors for modeling.
        - The score is a weighted sum; higher values indicate higher risk.

    Return schema:
        DataFrame with all original columns plus [new_col] (float).
    """
    required_cols = [favc_col, caec_col, calc_col, scc_col, smoke_col,
                     fcvc_col, ch2o_col, faf_col, tue_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    df = df.copy()
    # Map yes/no to 1/0
    yes_no_map = {'yes': 1, 'no': 0}
    favc = df[favc_col].map(yes_no_map)
    scc = df[scc_col].map(yes_no_map)
    smoke = df[smoke_col].map(yes_no_map)

    # Map CAEC and CALC to ordinal values
    caec_map = {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    calc_map = {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}
    caec = df[caec_col].str.capitalize().map(caec_map)
    calc = df[calc_col].str.capitalize().map(calc_map)

    # Normalize positive behaviors (higher is better, so negative risk)
    fcvc = df[fcvc_col].fillna(df[fcvc_col].median())
    ch2o = df[ch2o_col].fillna(df[ch2o_col].median())
    faf = df[faf_col].fillna(df[faf_col].median())
    tue = df[tue_col].fillna(df[tue_col].median())

    # Scale positive behaviors (reverse sign for risk)
    fcvc_scaled = -zscore(fcvc)
    ch2o_scaled = -zscore(ch2o)
    faf_scaled = -zscore(faf)
    tue_scaled = zscore(tue)

    score = (
        1.0 * favc.fillna(0) +
        0.7 * caec.fillna(0) +
        0.5 * calc.fillna(0) +
        0.8 * smoke.fillna(0) +
        0.8 * (1 - scc.fillna(0)) +  # Not monitoring calories increases risk
        0.5 * fcvc_scaled +
        0.5 * ch2o_scaled +
        0.7 * faf_scaled +
        0.7 * tue_scaled
    )
    df[new_col] = score
    return df

def create_age_lifestyle_interaction(
    df: pd.DataFrame,
    age_col: str = "Age",
    favc_col: str = "FAVC",
    fcvc_col: str = "FCVC",
    faf_col: str = "FAF",
    new_col: str = "age_lifestyle_interaction"
) -> pd.DataFrame:
    """
    Create an interaction feature between age and composite lifestyle factors.

    Args:
        df (pd.DataFrame): Input DataFrame.
        age_col (str): Column for age.
        favc_col (str): High-caloric food consumption (yes/no).
        fcvc_col (str): Vegetable consumption frequency.
        faf_col (str): Physical activity frequency.
        new_col (str): Name for the new interaction column.

    Returns:
        pd.DataFrame: DataFrame with the new interaction feature.

    Raises:
        ValueError: If any required column is missing.

    Notes:
        - Dataset-specific tool.
        - Domain rationale: Age moderates the impact of lifestyle; this feature captures non-linear risk growth with unhealthy behaviors as age increases.

    Return schema:
        DataFrame with original columns plus [new_col] (float).
    """
    required_cols = [age_col, favc_col, fcvc_col, faf_col]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    df = df.copy()
    favc_num = df[favc_col].map({'yes': 1, 'no': 0})
    fcvc = df[fcvc_col].fillna(df[fcvc_col].median())
    faf = df[faf_col].fillna(df[faf_col].median())
    # Higher interaction = older age, more high-cal food, less veggies, less activity
    interaction = df[age_col] * (favc_num + (fcvc.max() - fcvc) + (faf.max() - faf))
    df[new_col] = interaction
    return df

def create_bmi_weight_interaction(
    df: pd.DataFrame,
    bmi_col: str = "BMI",
    weight_col: str = "Weight",
    new_col: str = "bmi_weight_interaction"
) -> pd.DataFrame:
    """
    Create an interaction feature between BMI and Weight.

    Args:
        df (pd.DataFrame): Input DataFrame.
        bmi_col (str): BMI column.
        weight_col (str): Weight column.
        new_col (str): Name for the new interaction column.

    Returns:
        pd.DataFrame: DataFrame with new interaction column.

    Raises:
        ValueError: If any required column is missing.

    Notes:
        - Dataset-specific tool.
        - Domain rationale: High collinearity between BMI and Weight; this term may help models capture non-linear relationships or flag redundant information.

    Return schema:
        DataFrame with original columns plus [new_col] (float).
    """
    if bmi_col not in df.columns or weight_col not in df.columns:
        raise ValueError(f"Columns '{bmi_col}' and/or '{weight_col}' not found in DataFrame.")
    df = df.copy()
    df[new_col] = df[bmi_col] * df[weight_col]
    return df

def create_lifestyle_balance_feature(
    df: pd.DataFrame,
    faf_col: str = "FAF",
    tue_col: str = "TUE",
    new_col: str = "lifestyle_balance"
) -> pd.DataFrame:
    """
    Create a feature representing the balance between physical activity and sedentary behavior.

    Args:
        df (pd.DataFrame): Input DataFrame.
        faf_col (str): Physical activity frequency.
        tue_col (str): Time using technology devices.
        new_col (str): Name for new balance column.

    Returns:
        pd.DataFrame: DataFrame with new lifestyle balance column.

    Raises:
        ValueError: If required columns are missing.

    Notes:
        - Dataset-specific tool.
        - Domain rationale: Obesity risk is affected by the balance of activity (FAF) and sedentary (TUE) time.

    Return schema:
        DataFrame with original columns plus [new_col] (float).
    """
    if faf_col not in df.columns or tue_col not in df.columns:
        raise ValueError(f"Columns '{faf_col}' and/or '{tue_col}' not found in DataFrame.")
    df = df.copy()
    # Positive values: more activity relative to sedentary, negative: more sedentary
    df[new_col] = df[faf_col] - df[tue_col]
    return df

# ===============================
# == GENERIC (REUSABLE) TOOLS ===
# ===============================

def standardize_numerical_features(
    df: pd.DataFrame,
    numerical_cols: List[str],
    suffix: str = "_std"
) -> pd.DataFrame:
    """
    Standardize numerical features using z-score.

    Args:
        df (pd.DataFrame): Input DataFrame.
        numerical_cols (List[str]): List of numerical columns to standardize.
        suffix (str): Suffix to append to standardized columns.

    Returns:
        pd.DataFrame: DataFrame with additional standardized columns.

    Raises:
        ValueError: If any column not in df.

    Notes:
        - Generic tool.
        - Missing values are ignored in standardization but columns will still be present.

    Return schema:
        DataFrame with original columns plus [col+suffix] for each col in numerical_cols.
    """
    for col in numerical_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    df = df.copy()
    for col in numerical_cols:
        df[f"{col}{suffix}"] = zscore(df[col].astype(float), nan_policy='omit')
    return df

def one_hot_encode_columns(
    df: pd.DataFrame,
    categorical_cols: List[str],
    drop_original: bool = False,
    prefix_sep: str = "_"
) -> pd.DataFrame:
    """
    One-hot encode specified categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (List[str]): List of categorical columns to encode.
        drop_original (bool): If True, drop the original categorical columns.
        prefix_sep (str): Separator for new column names.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns.

    Raises:
        ValueError: If any column not in df.

    Notes:
        - Generic tool.
        - All categories (including rare) are encoded.

    Return schema:
        DataFrame with one-hot encoded columns and optionally original columns dropped.
    """
    for col in categorical_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")

    df = df.copy()
    ohe_df = pd.get_dummies(df[categorical_cols], prefix=categorical_cols, prefix_sep=prefix_sep)
    df = pd.concat([df, ohe_df], axis=1)
    if drop_original:
        df = df.drop(columns=categorical_cols)
    return df

def pca_feature_reduction(
    df: pd.DataFrame,
    feature_cols: List[str],
    n_components: int = 2,
    prefix: str = "PCA_"
) -> pd.DataFrame:
    """
    Apply Principal Component Analysis (PCA) to numerical features.

    Args:
        df (pd.DataFrame): Input DataFrame.
        feature_cols (List[str]): List of columns to include in PCA.
        n_components (int): Number of principal components to extract.
        prefix (str): Prefix for PCA component columns.

    Returns:
        pd.DataFrame: DataFrame with new PCA component columns.

    Raises:
        ValueError: If any column not in df, or n_components invalid.

    Notes:
        - Generic tool.
        - PCA is fit on input data only.

    Return schema:
        DataFrame with original columns plus [prefix+'1', ..., prefix+str(n_components)].
    """
    for col in feature_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    if n_components < 1 or n_components > len(feature_cols):
        raise ValueError("n_components must be between 1 and number of feature_cols.")

    df = df.copy()
    X = df[feature_cols].astype(float).fillna(df[feature_cols].mean())
    pca = PCA(n_components=n_components, random_state=42)
    pcs = pca.fit_transform(X)
    for i in range(n_components):
        df[f"{prefix}{i+1}"] = pcs[:, i]
    return df

def aggregate_flag_features(
    df: pd.DataFrame,
    flag_cols: List[str],
    new_col: str = "n_flags"
) -> pd.DataFrame:
    """
    Aggregate multiple boolean flag columns into a single count feature.

    Args:
        df (pd.DataFrame): Input DataFrame.
        flag_cols (List[str]): List of boolean flag columns.
        new_col (str): Name for the aggregated count column.

    Returns:
        pd.DataFrame: DataFrame with aggregated flag count column.

    Raises:
        ValueError: If any column not in df.

    Notes:
        - Generic tool.
        - Useful for summarizing data quality or outlier flags.

    Return schema:
        DataFrame with original columns plus [new_col] (int).
    """
    for col in flag_cols:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    df = df.copy()
    df[new_col] = df[flag_cols].sum(axis=1)
    return df

def target_mean_encoding(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: str,
    min_samples_leaf: int = 1,
    smoothing: float = 1.0,
    suffix: str = "_target_enc"
) -> pd.DataFrame:
    """
    Perform target mean encoding for categorical columns (suitable for train only).

    Args:
        df (pd.DataFrame): Input DataFrame.
        categorical_cols (List[str]): Categorical columns to encode.
        target_col (str): Target variable (must be numeric or ordinal encoded).
        min_samples_leaf (int): Minimum samples to allow for regularization.
        smoothing (float): Smoothing strength (higher = more global mean).
        suffix (str): Suffix for new encoded columns.

    Returns:
        pd.DataFrame: DataFrame with new target-encoded columns.

    Raises:
        ValueError: If any column not in df, or if target_col is not present.

    Notes:
        - Generic tool, but must be applied with care (leakage risk).
        - For multiclass targets, must provide ordinal/numeric encoding.
        - Should be fit on train, then mapping applied to test.

    Return schema:
        DataFrame with original columns plus [col+suffix] for each col in categorical_cols (float).
    """
    for col in categorical_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    df = df.copy()
    global_mean = df[target_col].mean()
    for col in categorical_cols:
        stats = df.groupby(col)[target_col].agg(['mean', 'count'])
        smoothing_val = 1 / (1 + np.exp(-(stats['count'] - min_samples_leaf) / smoothing))
        enc_map = global_mean * (1 - smoothing_val) + stats['mean'] * smoothing_val
        df[f"{col}{suffix}"] = df[col].map(enc_map)
    return df

# ===============================
# == END OF MODULE ==
# ===============================