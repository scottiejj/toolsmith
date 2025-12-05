import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional, Union
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

#########################
# PRELIMINARY EDA TOOLS #
#########################

def summarize_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Summarizes categorical columns: unique values, value counts, and missing values.
    
    Dataset-specific or Generic: Generic

    Args:
        df (pd.DataFrame): The data frame containing the data.
        categorical_cols (List[str]): List of column names to summarize.

    Returns:
        pd.DataFrame: Summary table with columns: 'feature', 'num_unique', 'most_common', 'most_common_freq', 'missing_count'.

    Raises:
        ValueError: If any of the categorical columns are missing in the DataFrame.

    Example:
        >>> summarize_categorical_features(df, ['Gender', 'SMOKE'])
    """
    missing = [col for col in categorical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    summary = []
    for col in categorical_cols:
        vc = df[col].value_counts(dropna=False)
        most_common = vc.index[0] if not vc.empty else np.nan
        most_common_freq = vc.iloc[0] if not vc.empty else 0
        summary.append({
            'feature': col,
            'num_unique': df[col].nunique(dropna=True),
            'most_common': most_common,
            'most_common_freq': most_common_freq,
            'missing_count': df[col].isnull().sum()
        })
    return pd.DataFrame(summary)

def summarize_numerical_features(
    df: pd.DataFrame,
    numerical_cols: List[str]
) -> pd.DataFrame:
    """
    Provides descriptive statistics for numerical columns.
    
    Dataset-specific or Generic: Generic

    Args:
        df (pd.DataFrame): The data frame containing the data.
        numerical_cols (List[str]): List of numerical column names.

    Returns:
        pd.DataFrame: Descriptive statistics table with columns: 'feature', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_count'.

    Raises:
        ValueError: If any of the numerical columns are missing in the DataFrame.

    Example:
        >>> summarize_numerical_features(df, ['Age', 'Height'])
    """
    missing = [col for col in numerical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    desc = df[numerical_cols].describe().T
    desc['missing_count'] = df[numerical_cols].isnull().sum()
    columns = ['mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_count']
    desc = desc.rename_axis('feature').reset_index()
    return desc[['feature'] + [c for c in columns if c in desc.columns]]

def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_cols: List[str],
    target_col: Optional[str] = None,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Plots histograms (optionally colored by target class) for multiple numerical features.

    Dataset-specific or Generic: Generic

    Args:
        df (pd.DataFrame): DataFrame with data.
        numerical_cols (List[str]): List of numerical features to plot.
        target_col (Optional[str]): If provided, use as hue for color separation.
        n_cols (int): Number of columns in subplot grid.
        figsize (Tuple[int, int]): Figure size.

    Returns:
        plt.Figure: The matplotlib Figure object (not shown).

    Raises:
        ValueError: If any numerical columns are missing.

    Example:
        >>> fig = plot_numerical_distributions(df, ['Age', 'Weight'], target_col='NObeyesdad')
    """
    missing = [col for col in numerical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    n = len(numerical_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows))
    axs = axs.flatten()
    for i, col in enumerate(numerical_cols):
        ax = axs[i]
        if target_col and target_col in df.columns:
            sns.histplot(data=df, x=col, hue=target_col, multiple='stack', ax=ax, kde=False)
        else:
            sns.histplot(df[col].dropna(), ax=ax, kde=True)
        ax.set_title(col)
    # Remove unused subplots
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    fig.tight_layout()
    return fig

def plot_categorical_counts(
    df: pd.DataFrame,
    categorical_cols: List[str],
    target_col: Optional[str] = None,
    n_cols: int = 3,
    figsize: Tuple[int, int] = (16, 4)
) -> plt.Figure:
    """
    Plots countplots for multiple categorical features (optionally colored by target class).

    Dataset-specific or Generic: Generic

    Args:
        df (pd.DataFrame): DataFrame with data.
        categorical_cols (List[str]): Categorical features to plot.
        target_col (Optional[str]): Target column for hue grouping.
        n_cols (int): Number of columns in subplot grid.
        figsize (Tuple[int, int]): Figure size.

    Returns:
        plt.Figure: The matplotlib Figure object (not shown).

    Raises:
        ValueError: If any categorical columns are missing.

    Example:
        >>> fig = plot_categorical_counts(df, ['Gender', 'SMOKE'], target_col='NObeyesdad')
    """
    missing = [col for col in categorical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    n = len(categorical_cols)
    n_rows = (n + n_cols - 1) // n_cols
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(figsize[0], figsize[1]*n_rows))
    axs = axs.flatten()
    for i, col in enumerate(categorical_cols):
        ax = axs[i]
        if target_col and target_col in df.columns:
            sns.countplot(data=df, x=col, hue=target_col, ax=ax)
        else:
            sns.countplot(data=df, x=col, ax=ax)
        ax.set_title(col)
        ax.legend(loc='best', fontsize='small') if target_col else None
    for j in range(i+1, len(axs)):
        fig.delaxes(axs[j])
    fig.tight_layout()
    return fig

def analyze_target_distribution(
    df: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    """
    Analyzes the class distribution of the target variable.

    Dataset-specific or Generic: Dataset-specific

    Domain rationale:
        The dataset's target is a multi-class categorical variable ("NObeyesdad") with potentially imbalanced classes.
        Understanding class distribution is crucial for modeling and evaluation.

    Args:
        df (pd.DataFrame): DataFrame with target column.
        target_col (str): The target column name.

    Returns:
        pd.DataFrame: Table with 'class', 'count', 'proportion'.

    Raises:
        ValueError: If target_col is missing.

    Example:
        >>> analyze_target_distribution(df, 'NObeyesdad')
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in DataFrame.")
    vc = df[target_col].value_counts(dropna=False)
    total = len(df)
    result = pd.DataFrame({
        'class': vc.index,
        'count': vc.values,
        'proportion': vc.values / total
    })
    return result

def correlation_matrix_heatmap(
    df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = "pearson",
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plots a correlation matrix heatmap for numerical features.

    Dataset-specific or Generic: Generic

    Args:
        df (pd.DataFrame): DataFrame.
        numerical_cols (List[str]): List of numerical columns to include.
        method (str): Correlation method ('pearson', 'spearman', 'kendall').
        figsize (Tuple[int, int]): Figure size.

    Returns:
        plt.Figure: The matplotlib Figure object (not shown).

    Raises:
        ValueError: If any numerical columns are missing or method is invalid.

    Example:
        >>> fig = correlation_matrix_heatmap(df, ['Age', 'Height', 'Weight'])
    """
    missing = [col for col in numerical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    if method not in ['pearson', 'spearman', 'kendall']:
        raise ValueError("method must be one of: 'pearson', 'spearman', 'kendall'")
    corr = df[numerical_cols].corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    ax.set_title(f"{method.title()} Correlation Matrix")
    return fig

def bmi_feature_analysis(
    df: pd.DataFrame,
    height_col: str,
    weight_col: str,
    target_col: Optional[str] = None
) -> Dict[str, Union[pd.Series, plt.Figure]]:
    """
    Computes BMI (Body Mass Index) for each individual, summarizes its distribution, and optionally visualizes by target class.

    Dataset-specific or Generic: Dataset-specific

    Domain rationale:
        BMI is a primary indicator for obesity risk, directly related to the prediction target.
        Assessing its distribution and its relation to "NObeyesdad" provides critical insights.

    Args:
        df (pd.DataFrame): Input DataFrame.
        height_col (str): Column for height (meters).
        weight_col (str): Column for weight (kg).
        target_col (Optional[str]): If provided, produces a plot colored by this column.

    Returns:
        Dict[str, Union[pd.Series, plt.Figure]]:
            {
                'bmi_series': pd.Series (BMI values, indexed as df),
                'bmi_summary': pd.Series (basic statistics),
                'bmi_plot': plt.Figure (if target_col provided, else None)
            }

    Raises:
        ValueError: If columns are missing or contain invalid (<=0) values.

    Example:
        >>> result = bmi_feature_analysis(df, 'Height', 'Weight', target_col='NObeyesdad')
        >>> result['bmi_series']
        >>> result['bmi_summary']
        >>> fig = result['bmi_plot']
    """
    for col in [height_col, weight_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    # Avoid division by zero or negative values
    height = df[height_col].astype(float)
    weight = df[weight_col].astype(float)
    if (height <= 0).any():
        raise ValueError("All height values must be positive and non-zero.")
    bmi = weight / (height ** 2)
    bmi_summary = bmi.describe()
    fig = None
    if target_col and target_col in df.columns:
        fig, ax = plt.subplots(figsize=(8,5))
        sns.histplot(data=df.assign(BMI=bmi), x='BMI', hue=target_col, multiple='stack', kde=True, ax=ax)
        ax.set_title('BMI Distribution by ' + target_col)
    return {
        'bmi_series': bmi,
        'bmi_summary': bmi_summary,
        'bmi_plot': fig
    }

def automated_outlier_detector(
    df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = "iqr",
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Identifies outliers in numerical columns using IQR or z-score method.

    Dataset-specific or Generic: Generic

    Args:
        df (pd.DataFrame): DataFrame to check.
        numerical_cols (List[str]): List of numerical columns to check.
        method (str): 'iqr' or 'zscore'.
        threshold (float): Outlier threshold (IQR multiple or z-score).

    Returns:
        pd.DataFrame: Table with columns: 'feature', 'outlier_count', 'outlier_ratio', 'method', 'threshold'.

    Raises:
        ValueError: If any columns are missing, or method invalid.

    Example:
        >>> automated_outlier_detector(df, ['Age', 'Weight'])
    """
    missing = [col for col in numerical_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    if method not in ["iqr", "zscore"]:
        raise ValueError("method must be 'iqr' or 'zscore'")
    results = []
    for col in numerical_cols:
        series = df[col].dropna()
        if method == "iqr":
            Q1 = series.quantile(0.25)
            Q3 = series.quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - threshold * IQR
            upper = Q3 + threshold * IQR
            mask = (series < lower) | (series > upper)
        else:  # zscore
            mean = series.mean()
            std = series.std()
            if std == 0:
                mask = pd.Series([False]*len(series), index=series.index)
            else:
                zscores = (series - mean) / std
                mask = (zscores < -threshold) | (zscores > threshold)
        outlier_count = mask.sum()
        results.append({
            'feature': col,
            'outlier_count': int(outlier_count),
            'outlier_ratio': float(outlier_count) / len(series) if len(series) > 0 else np.nan,
            'method': method,
            'threshold': threshold
        })
    return pd.DataFrame(results)

def target_feature_impact_analysis(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    feature_types: Optional[Dict[str, str]] = None
) -> pd.DataFrame:
    """
    Quantifies the association between features and the target.
    For numerical features: computes ANOVA F-statistic and p-value.
    For categorical features: computes Cramér's V.

    Dataset-specific or Generic: Dataset-specific

    Domain rationale:
        The target is multi-class categorical, so understanding which features (diet, lifestyle, etc.)
        show the strongest association with obesity risk is critical for both feature selection and interpretation.

    Args:
        df (pd.DataFrame): DataFrame.
        feature_cols (List[str]): Features to analyze.
        target_col (str): The target variable (must be categorical).
        feature_types (Optional[Dict[str, str]]): Dict mapping feature name to 'categorical' or 'numerical'.
                                                 If None, types are inferred.

    Returns:
        pd.DataFrame: Table with columns:
            'feature', 'feature_type', 'association', 'p_value' (for numerical), 'method'

    Raises:
        ValueError: If columns are missing.

    Example:
        >>> target_feature_impact_analysis(df, ['Age', 'Gender', 'FAF'], 'NObeyesdad')
    """
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found.")
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Feature columns not found: {missing}")
    # Infer types if not provided
    inferred_types = {}
    for col in feature_cols:
        if feature_types and col in feature_types:
            inferred_types[col] = feature_types[col]
        else:
            # Heuristic: object or category = categorical, else numerical
            dtype = df[col].dtype
            if pd.api.types.is_numeric_dtype(dtype):
                inferred_types[col] = 'numerical'
            else:
                inferred_types[col] = 'categorical'
    results = []
    for col in feature_cols:
        ftype = inferred_types[col]
        if ftype == 'numerical':
            # One-way ANOVA F-test
            groups = [df[df[target_col]==cls][col].dropna() for cls in df[target_col].unique()]
            if any(len(g)==0 for g in groups):
                # If any class is empty, skip
                association, pval = np.nan, np.nan
            else:
                association, pval = stats.f_oneway(*groups)
            method = 'ANOVA F-stat'
        else:
            # Categorical: Cramér's V
            confusion = pd.crosstab(df[col], df[target_col])
            chi2 = stats.chi2_contingency(confusion, correction=False)[0]
            n = confusion.sum().sum()
            k = min(confusion.shape)  # min(num categories, num classes)
            # Avoid divide by zero
            cramers_v = np.sqrt(chi2 / (n * (k-1))) if k > 1 and n > 0 else np.nan
            association, pval = cramers_v, np.nan
            method = "Cramér's V"
        results.append({
            'feature': col,
            'feature_type': ftype,
            'association': association,
            'p_value': pval,
            'method': method
        })
    return pd.DataFrame(results)

def feature_vs_target_boxplot(
    df: pd.DataFrame,
    numerical_col: str,
    target_col: str,
    figsize: Tuple[int, int] = (8,5)
) -> plt.Figure:
    """
    Boxplot of a numerical feature grouped by target classes.

    Dataset-specific or Generic: Dataset-specific

    Domain rationale:
        For this dataset, visualizing how BMI, Age, Weight, or other numerics vary by obesity class can reveal
        separable patterns and inform feature engineering.

    Args:
        df (pd.DataFrame): DataFrame.
        numerical_col (str): Feature to plot.
        target_col (str): Target variable.

    Returns:
        plt.Figure: The matplotlib Figure object.

    Raises:
        ValueError: If columns are missing.

    Example:
        >>> fig = feature_vs_target_boxplot(df, 'Age', 'NObeyesdad')
    """
    for col in [numerical_col, target_col]:
        if col not in df.columns:
            raise ValueError(f"Column '{col}' not found in DataFrame.")
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=target_col, y=numerical_col, data=df, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_title(f'{numerical_col} by {target_col}')
    return fig

####################
# END OF MODULE    #
####################