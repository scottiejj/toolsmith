import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Any, Union
from scipy.stats import zscore

###############################################################################
# Data Cleaning Tools for the "obesity_risks" Dataset
#
# This module contains both dataset-specific and general-purpose data cleaning
# tools. Each function is pure, parameterized, and does not perform any file I/O.
###############################################################################

def bmi_validator_and_flagger(
    df: pd.DataFrame,
    height_col: str,
    weight_col: str,
    bmi_col: str = "BMI",
    flag_col: str = "BMI_flag"
) -> pd.DataFrame:
    """
    [DATASET-SPECIFIC]
    Calculate BMI, flag implausible values, and return the modified DataFrame.

    Domain rationale:
        BMI is central to obesity classification. Height (in meters) and weight (in kg)
        allow calculation of BMI. Medical literature considers BMI < 10 or > 80 implausible.
        Flagging these helps identify data entry or synthetic errors unique to this domain.

    Args:
        df (pd.DataFrame): Input DataFrame.
        height_col (str): Name of the height column (must be in meters).
        weight_col (str): Name of the weight column (must be in kilograms).
        bmi_col (str, optional): Name for the new BMI column. Defaults to "BMI".
        flag_col (str, optional): Name for the new flag column. Defaults to "BMI_flag".

    Returns:
        pd.DataFrame: Copy of input with two new columns: BMI (float), BMI_flag (bool).
            Return schema:
                - All original columns
                - `{bmi_col}`: float, calculated BMI
                - `{flag_col}`: bool, True if BMI < 10 or BMI > 80

    Raises:
        ValueError: If columns are missing or contain non-positive values.

    Example:
        out = bmi_validator_and_flagger(df, "Height", "Weight")
        flagged = out[out["BMI_flag"]]
    """
    if height_col not in df.columns or weight_col not in df.columns:
        raise ValueError(f"Missing required columns: '{height_col}' or '{weight_col}'")
    if not np.issubdtype(df[height_col].dtype, np.number) or not np.issubdtype(df[weight_col].dtype, np.number):
        raise TypeError("Height and weight columns must be numeric.")
    if (df[height_col] <= 0).any():
        raise ValueError("Height column contains non-positive values, cannot compute BMI.")
    df_ = df.copy()
    df_[bmi_col] = df_[weight_col] / (df_[height_col] ** 2)
    df_[flag_col] = (df_[bmi_col] < 10) | (df_[bmi_col] > 80)
    return df_

def handle_rare_categories(
    df: pd.DataFrame,
    col: str,
    threshold: int = 10,
    new_category: str = "Rare"
) -> pd.DataFrame:
    """
    [GENERIC]
    Replace rare categories in a categorical column with a new label.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Categorical column to process.
        threshold (int): Categories with count <= threshold are replaced. Default 10.
        new_category (str): The label to use for rare categories. Default "Rare".

    Returns:
        pd.DataFrame: Copy of input with rare categories replaced.
            Return schema:
                - All original columns
                - `{col}`: with rare categories replaced by `new_category`.

    Raises:
        ValueError: If col is missing.

    Notes:
        The function is generic and can be used for any categorical variable.
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    counts = df[col].value_counts()
    to_replace = counts[counts <= threshold].index
    df_ = df.copy()
    df_[col] = df_[col].replace(to_replace, new_category)
    return df_

def ordinal_categorical_cleaner(
    df: pd.DataFrame,
    col: str,
    valid_order: List[str],
    fill_value: Optional[str] = None
) -> pd.DataFrame:
    """
    [DATASET-SPECIFIC]
    Fix inconsistencies and optionally impute missing/invalid values in ordinal categorical columns.

    Domain rationale:
        Features like CAEC and CALC have expected categories (e.g., ['no', 'Sometimes', 'Frequently', 'Always']).
        Synthetic or entry errors may create typos or missing values. Standardizing values and filling missing/invalids
        helps maintain ordinal relationships crucial for this dataset.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Ordinal categorical column to clean.
        valid_order (List[str]): List of valid ordered categories.
        fill_value (Optional[str]): Value to use for missing/invalid entries.
            If None, invalids are kept as is.

    Returns:
        pd.DataFrame: Copy with cleaned column.
            Return schema:
                - All original columns
                - `{col}` is cast as pd.Categorical (ordered=True) with given categories.

    Raises:
        ValueError: If col is missing or valid_order is empty.

    Example:
        cleaned = ordinal_categorical_cleaner(df, "CAEC", ["no", "Sometimes", "Frequently", "Always"], fill_value="no")
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not valid_order:
        raise ValueError("valid_order must be a non-empty list of categories.")
    df_ = df.copy()
    fixed = df_[col].where(df_[col].isin(valid_order), fill_value)
    cat_dtype = pd.CategoricalDtype(categories=valid_order, ordered=True)
    df_[col] = pd.Series(fixed, dtype=cat_dtype)
    return df_

def categorical_binary_cleaner(
    df: pd.DataFrame,
    col: str,
    valid_values: Tuple[str, str] = ("yes", "no"),
    fill_value: Optional[str] = None,
    case_insensitive: bool = True
) -> pd.DataFrame:
    """
    [DATASET-SPECIFIC]
    Standardize binary categorical columns to 'yes'/'no' and optionally impute/fix invalids.

    Domain rationale:
        Most binary columns in this dataset use 'yes'/'no'. Ensuring consistent casing and handling
        unexpected categories reduces noise, especially for features like 'family_history_with_overweight'.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Column to standardize.
        valid_values (Tuple[str, str]): Acceptable binary values. Defaults to ('yes', 'no').
        fill_value (Optional[str]): Value for missing/invalids. If None, leaves as is.
        case_insensitive (bool): If True, matches regardless of case.

    Returns:
        pd.DataFrame: Copy with standardized column.
            Return schema:
                - All original columns
                - `{col}`: standardized with only valid_values + fill_value.

    Raises:
        ValueError: If col missing or valid_values not length 2.

    Example:
        df2 = categorical_binary_cleaner(df, "FAVC", ("yes", "no"), fill_value="no")
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if len(valid_values) != 2:
        raise ValueError("valid_values must be a tuple of length 2.")
    df_ = df.copy()
    if case_insensitive:
        series = df_[col].astype(str).str.lower()
        valid_map = {v.lower(): v for v in valid_values}
        fixed = series.map(valid_map).where(series.isin(valid_map), fill_value)
    else:
        fixed = df_[col].where(df_[col].isin(valid_values), fill_value)
    df_[col] = fixed
    return df_

def iqr_outlier_flagger(
    df: pd.DataFrame,
    col: str,
    iqr_mult: float = 1.5,
    flag_col: Optional[str] = None
) -> pd.DataFrame:
    """
    [GENERIC]
    Flag outliers in a numerical column using the IQR method.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Numerical column to assess.
        iqr_mult (float): Outlier threshold multiplier. Default 1.5.
        flag_col (Optional[str]): Name for the flag column. If None, uses '{col}_outlier_flag'.

    Returns:
        pd.DataFrame: Copy with additional boolean flag column.
            Return schema:
                - All original columns
                - `{flag_col}`: bool, True if value is an outlier.

    Raises:
        ValueError: If col is missing or not numeric.

    Example:
        out = iqr_outlier_flagger(df, "Age")
        df_clean = out[~out["Age_outlier_flag"]]
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if not np.issubdtype(df[col].dtype, np.number):
        raise TypeError(f"Column '{col}' must be numeric.")
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - iqr_mult * iqr
    upper = q3 + iqr_mult * iqr
    flag = (df[col] < lower) | (df[col] > upper)
    flag_col = flag_col or f"{col}_outlier_flag"
    df_ = df.copy()
    df_[flag_col] = flag
    return df_

def numerical_type_enforcer(
    df: pd.DataFrame,
    cols: List[str],
    coerce: bool = True,
    fill_invalid_with: Optional[float] = None
) -> pd.DataFrame:
    """
    [GENERIC]
    Enforce numerical dtype on specified columns, optionally coercing non-numeric entries.

    Args:
        df (pd.DataFrame): Input DataFrame.
        cols (List[str]): List of column names to convert.
        coerce (bool): If True, converts errors to NaN.
        fill_invalid_with (Optional[float]): If set, fills NaN with this value after conversion.

    Returns:
        pd.DataFrame: Copy with converted columns.
            Return schema:
                - All original columns (specified cols are float).

    Raises:
        ValueError: If columns are missing.

    Example:
        fixed = numerical_type_enforcer(df, ["Age", "Weight"], fill_invalid_with=0)
    """
    df_ = df.copy()
    for c in cols:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not found in DataFrame.")
        df_[c] = pd.to_numeric(df_[c], errors='coerce' if coerce else 'raise')
        if fill_invalid_with is not None:
            df_[c].fillna(fill_invalid_with, inplace=True)
    return df_

def gender_cleaner(
    df: pd.DataFrame,
    col: str = "Gender",
    valid_values: Tuple[str, str] = ("Male", "Female"),
    fill_value: Optional[str] = None
) -> pd.DataFrame:
    """
    [DATASET-SPECIFIC]
    Standardize the Gender column to 'Male'/'Female', fixing case and typos.

    Domain rationale:
        Gender is a key demographic variable. Ensuring only 'Male'/'Female'
        values, regardless of case/typos, is important for clarity and modeling.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Gender column name.
        valid_values (Tuple[str, str]): Acceptable gender values.
        fill_value (Optional[str]): Value for missing/invalids.

    Returns:
        pd.DataFrame: Copy with cleaned gender column.
            Return schema:
                - All original columns
                - `{col}`: standardized to valid_values + fill_value.

    Raises:
        ValueError: If col missing or valid_values not length 2.

    Example:
        df2 = gender_cleaner(df, "Gender")
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    if len(valid_values) != 2:
        raise ValueError("valid_values must be a tuple of length 2.")
    df_ = df.copy()
    gender_map = {v.lower(): v for v in valid_values}
    fixed = df_[col].astype(str).str.lower().map(gender_map).where(
        df_[col].astype(str).str.lower().isin(gender_map), fill_value
    )
    df_[col] = fixed
    return df_

def drop_uninformative_id_column(
    df: pd.DataFrame,
    id_col: str = "id"
) -> pd.DataFrame:
    """
    [GENERIC]
    Drop ID column if present (for use before model training, not during submission prep).

    Args:
        df (pd.DataFrame): Input DataFrame.
        id_col (str): Name of the ID column.

    Returns:
        pd.DataFrame: Copy without ID column if present.

    Raises:
        None

    Notes:
        Does nothing if `id_col` not present.
    """
    df_ = df.copy()
    if id_col in df_:
        df_.drop(columns=[id_col], inplace=True)
    return df_

def impute_missing_by_group_median(
    df: pd.DataFrame,
    target_col: str,
    group_col: str
) -> pd.DataFrame:
    """
    [GENERIC]
    Impute missing values in a numerical column using the median within groups.

    Args:
        df (pd.DataFrame): Input DataFrame.
        target_col (str): Numerical column to impute.
        group_col (str): Categorical column to group by.

    Returns:
        pd.DataFrame: Copy with missing values in target_col imputed.
            Return schema:
                - All original columns (target_col filled as float).

    Raises:
        ValueError: If target_col or group_col not found.

    Example:
        df2 = impute_missing_by_group_median(df, "Weight", "Gender")
    """
    if target_col not in df.columns or group_col not in df.columns:
        raise ValueError(f"Columns '{target_col}' or '{group_col}' not found in DataFrame.")
    df_ = df.copy()
    medians = df_.groupby(group_col)[target_col].transform("median")
    df_[target_col] = df_[target_col].fillna(medians)
    return df_

def cap_outliers(
    df: pd.DataFrame,
    col: str,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99
) -> pd.DataFrame:
    """
    [GENERIC]
    Cap values in a numerical feature at given lower/upper quantiles to address extreme outliers.

    Args:
        df (pd.DataFrame): Input DataFrame.
        col (str): Numerical column to cap.
        lower_quantile (float): Lower quantile (default 0.01).
        upper_quantile (float): Upper quantile (default 0.99).

    Returns:
        pd.DataFrame: Copy with capped column.
            Return schema:
                - All original columns, `{col}` capped.

    Raises:
        ValueError: If col not in DataFrame.

    Example:
        capped = cap_outliers(df, "Weight", 0.01, 0.99)
    """
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in DataFrame.")
    lower = df[col].quantile(lower_quantile)
    upper = df[col].quantile(upper_quantile)
    df_ = df.copy()
    df_[col] = df_[col].clip(lower, upper)
    return df_