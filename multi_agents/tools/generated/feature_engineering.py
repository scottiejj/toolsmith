import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from scipy.stats import zscore
from typing import List, Union, Dict, Any, Optional

def create_polynomial_features(df: pd.DataFrame, feature_cols: List[str], degree: int = 2) -> pd.DataFrame:
    """
    Generate polynomial features from the specified feature columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        feature_cols (List[str]): List of column names to generate polynomial features from.
        degree (int): The degree of the polynomial features.

    Returns:
        pd.DataFrame: DataFrame with polynomial features added.

    Raises:
        ValueError: If feature_cols are not found in the DataFrame.
    
    Examples:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> create_polynomial_features(df, ['A', 'B'], degree=2)
    """
    if not all(col in df.columns for col in feature_cols):
        raise ValueError(f"One or more columns in {feature_cols} are not in the DataFrame.")
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    poly_features = poly.fit_transform(df[feature_cols])
    poly_feature_names = poly.get_feature_names_out(feature_cols)
    
    return pd.DataFrame(poly_features, columns=poly_feature_names, index=df.index)

def one_hot_encode(df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
    """
    Perform one-hot encoding on specified categorical columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        categorical_cols (List[str]): List of column names to one-hot encode.

    Returns:
        pd.DataFrame: DataFrame with one-hot encoded columns added.

    Raises:
        ValueError: If categorical_cols are not found in the DataFrame.
    
    Examples:
        >>> df = pd.DataFrame({'Color': ['Red', 'Green'], 'Size': ['S', 'M']})
        >>> one_hot_encode(df, ['Color', 'Size'])
    """
    if not all(col in df.columns for col in categorical_cols):
        raise ValueError(f"One or more columns in {categorical_cols} are not in the DataFrame.")
    
    encoder = OneHotEncoder(sparse=False, drop='first')
    encoded_features = encoder.fit_transform(df[categorical_cols])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_cols), index=df.index)
    
    return pd.concat([df.drop(columns=categorical_cols), encoded_df], axis=1)

def fill_missing_values(df: pd.DataFrame, strategy: str = 'mean', columns: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Fill missing values in numerical columns using specified strategy.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        strategy (str): Strategy to use for filling missing values ('mean', 'median', or 'most_frequent').
        columns (Optional[List[str]]): Specific columns to fill. If None, fill all numeric columns.

    Returns:
        pd.DataFrame: DataFrame with missing values filled.

    Raises:
        ValueError: If strategy is not recognized or columns contain non-numeric types.
    
    Examples:
        >>> df = pd.DataFrame({'A': [1, np.nan, 3], 'B': [4, 5, np.nan]})
        >>> fill_missing_values(df, strategy='mean')
    """
    if strategy not in ['mean', 'median', 'most_frequent']:
        raise ValueError(f"Strategy '{strategy}' is not recognized. Use 'mean', 'median', or 'most_frequent'.")

    imputer = SimpleImputer(strategy=strategy)
    
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if not all(col in df.columns for col in columns):
        raise ValueError(f"One or more columns in {columns} are not in the DataFrame.")
    
    df[columns] = imputer.fit_transform(df[columns])
    return df

def zscore_normalization(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """
    Normalize specified columns using Z-score normalization.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        columns (List[str]): List of column names to normalize.

    Returns:
        pd.DataFrame: DataFrame with Z-score normalized columns.

    Raises:
        ValueError: If columns are not found in the DataFrame.
    
    Examples:
        >>> df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})
        >>> zscore_normalization(df, ['A'])
    """
    if not all(col in df.columns for col in columns):
        raise ValueError(f"One or more columns in {columns} are not in the DataFrame.")
    
    df[columns] = df[columns].apply(zscore)
    return df

def target_encoding(df: pd.DataFrame, target_col: str, categorical_cols: List[str], alpha: float = 0.1) -> pd.DataFrame:
    """
    Apply target encoding to categorical columns based on the target variable.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        target_col (str): The target column name.
        categorical_cols (List[str]): List of categorical column names to encode.
        alpha (float): Smoothing factor to prevent overfitting.

    Returns:
        pd.DataFrame: DataFrame with target encoded columns.

    Raises:
        ValueError: If target_col or categorical_cols are not found in the DataFrame.
    
    Examples:
        >>> df = pd.DataFrame({'Color': ['Red', 'Green', 'Red'], 'Target': [1, 0, 1]})
        >>> target_encoding(df, 'Target', ['Color'])
    """
    if target_col not in df.columns or not all(col in df.columns for col in categorical_cols):
        raise ValueError(f"Target column '{target_col}' or one of the categorical columns is not in the DataFrame.")
    
    mean_target = df[target_col].mean()
    
    encoding_map = df.groupby(categorical_cols)[target_col].mean().to_dict()
    
    for col in categorical_cols:
        df[f'{col}_encoded'] = df[col].map(lambda x: (encoding_map.get(x, mean_target) * (1 - alpha)) + 
                                             (mean_target * alpha))
    
    return df

def interaction_features(df: pd.DataFrame, feature_pairs: List[tuple]) -> pd.DataFrame:
    """
    Create interaction features from specified pairs of feature columns.

    Args:
        df (pd.DataFrame): Input DataFrame containing the data.
        feature_pairs (List[tuple]): List of tuples containing pairs of column names.

    Returns:
        pd.DataFrame: DataFrame with interaction features added.

    Raises:
        ValueError: If any feature pair contains columns not found in the DataFrame.
    
    Examples:
        >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> interaction_features(df, [('A', 'B')])
    """
    for col1, col2 in feature_pairs:
        if col1 not in df.columns or col2 not in df.columns:
            raise ValueError(f"Feature pair ({col1}, {col2}) contains columns not in the DataFrame.")
        
        df[f'{col1}_x_{col2}'] = df[col1] * df[col2]
    
    return df
