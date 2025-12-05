import pandas as pd
import numpy as np
from typing import List, Optional, Tuple, Dict, Union
import matplotlib
matplotlib.use('Agg')  # Prevents rendering during import
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency, f_oneway, kruskal, pearsonr, spearmanr


# ---------------------------
# Dataset-Specific EDA Tools
# ---------------------------

def segment_bmi_by_obesity_class(
    df: pd.DataFrame,
    bmi_col: str,
    target_col: str,
    outlier_flag_col: Optional[str] = None
) -> pd.DataFrame:
    """
    Segment BMI statistics by obesity class.

    Args:
        df (pd.DataFrame): Input dataframe.
        bmi_col (str): Name of the BMI column.
        target_col (str): Name of the target column (obesity class).
        outlier_flag_col (Optional[str]): Column indicating BMI outliers; if provided, outliers will be excluded.

    Returns:
        pd.DataFrame: BMI summary statistics (mean, std, min, max, median, count) for each obesity class.

    Raises:
        ValueError: If required columns are missing.

    Return schema:
        pd.DataFrame with columns:
            - target_col (obesity class/category)
            - mean
            - std
            - min
            - max
            - median
            - count

    Domain rationale:
        Obesity class is directly related to BMI. This function quantifies BMI characteristics per class to reveal separation, overlaps, or labeling inconsistencies.

    Example:
        >>> segment_bmi_by_obesity_class(train, 'BMI', 'NObeyesdad')
    """
    missing = [c for c in [bmi_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    df_ = df.copy()
    if outlier_flag_col:
        if outlier_flag_col not in df.columns:
            raise ValueError(f"Outlier flag column '{outlier_flag_col}' not in dataframe.")
        df_ = df_[~df_[outlier_flag_col].astype(bool)]
    stats = df_.groupby(target_col)[bmi_col].agg(['mean', 'std', 'min', 'max', 'median', 'count']).reset_index()
    return stats


def target_stratified_numerical_summary(
    df: pd.DataFrame,
    numerical_cols: List[str],
    target_col: str
) -> pd.DataFrame:
    """
    Compute summary statistics (mean, std, min, max, median) of numerical features stratified by target classes.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (List[str]): List of numerical columns to summarize.
        target_col (str): Name of the target column.

    Returns:
        pd.DataFrame: Multi-indexed summary statistics by target class.

    Raises:
        ValueError: If columns are missing.

    Return schema:
        pd.DataFrame with multi-index (target_col, stat), columns = numerical_cols

    Domain rationale:
        Obesity risk manifests differently across subgroups; stratified summaries reveal discriminative patterns and can guide feature engineering.

    Example:
        >>> target_stratified_numerical_summary(df, ['Age', 'Height', 'Weight'], 'NObeyesdad')
    """
    missing = [c for c in [target_col] + numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    grouped = df.groupby(target_col)[numerical_cols].agg(['mean', 'std', 'min', 'max', 'median', 'count'])
    # Flatten the column MultiIndex
    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()
    return grouped


def categorical_vs_target_chi2(
    df: pd.DataFrame,
    cat_cols: List[str],
    target_col: str
) -> pd.DataFrame:
    """
    Perform Chi-squared test between categorical features and the target variable.

    Args:
        df (pd.DataFrame): Input dataframe.
        cat_cols (List[str]): List of categorical columns.
        target_col (str): Name of the target column.

    Returns:
        pd.DataFrame: Results with columns ['feature', 'chi2', 'p_value', 'dof', 'significant'].

    Raises:
        ValueError: If columns are missing.

    Return schema:
        pd.DataFrame with columns:
            - feature (str)
            - chi2 (float)
            - p_value (float)
            - dof (int)
            - significant (bool, p<0.05)

    Domain rationale:
        Categorical features (e.g. family history, food habits) are hypothesized to be linked to obesity risk. Chi2 tests quantify dependence strength.

    Example:
        >>> categorical_vs_target_chi2(df, ['Gender', 'FAVC'], 'NObeyesdad')
    """
    missing = [c for c in [target_col] + cat_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    results = []
    for col in cat_cols:
        ct = pd.crosstab(df[col], df[target_col])
        if ct.shape[0] < 2 or ct.shape[1] < 2:
            results.append({'feature': col, 'chi2': np.nan, 'p_value': np.nan, 'dof': np.nan, 'significant': False})
            continue
        chi2, p, dof, _ = chi2_contingency(ct)
        results.append({'feature': col, 'chi2': chi2, 'p_value': p, 'dof': dof, 'significant': p < 0.05})
    return pd.DataFrame(results)


def target_vs_feature_anova(
    df: pd.DataFrame,
    numerical_cols: List[str],
    target_col: str
) -> pd.DataFrame:
    """
    Perform ANOVA/Kruskal-Wallis test for each numerical feature across target classes.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (List[str]): List of numerical columns.
        target_col (str): Name of the target column.

    Returns:
        pd.DataFrame: Test results for each feature.

    Raises:
        ValueError: If columns are missing.

    Return schema:
        pd.DataFrame with columns:
            - feature (str)
            - test (str, 'anova' or 'kruskal')
            - stat (float)
            - p_value (float)
            - significant (bool, p<0.05)

    Domain rationale:
        Quantifies whether means/medians of numerical features differ significantly between obesity classes.

    Example:
        >>> target_vs_feature_anova(df, ['Age', 'BMI'], 'NObeyesdad')
    """
    missing = [c for c in [target_col] + numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    results = []
    for col in numerical_cols:
        groups = [grp[1][col].dropna().values for grp in df.groupby(target_col)]
        if any(len(arr) < 2 for arr in groups):
            results.append({'feature': col, 'test': None, 'stat': np.nan, 'p_value': np.nan, 'significant': False})
            continue
        try:
            stat, p = f_oneway(*groups)
            test = 'anova'
        except Exception:
            stat, p = kruskal(*groups)
            test = 'kruskal'
        results.append({'feature': col, 'test': test, 'stat': stat, 'p_value': p, 'significant': p < 0.05})
    return pd.DataFrame(results)


def feature_interaction_heatmap(
    df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = 'pearson',
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Generate a heatmap of pairwise correlations (Pearson or Spearman) among numerical features.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (List[str]): List of numerical columns to correlate.
        method (str): Correlation method ('pearson' or 'spearman').
        figsize (Tuple[int, int]): Figure size.

    Returns:
        matplotlib.figure.Figure: The generated heatmap figure.

    Raises:
        ValueError: If columns are missing or method invalid.

    Return schema:
        matplotlib.figure.Figure

    Domain rationale:
        Identifies which health/lifestyle features are interdependent, possibly confounded, or redundant for modeling.

    Example:
        >>> fig = feature_interaction_heatmap(df, ['Age', 'Weight', 'BMI'])
    """
    if method not in {'pearson', 'spearman'}:
        raise ValueError("method must be 'pearson' or 'spearman'")
    missing = [c for c in numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")

    corr = df[numerical_cols].corr(method=method)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title(f'{method.capitalize()} Correlation Heatmap')
    return fig


def lifestyle_segment_obesity_rate(
    df: pd.DataFrame,
    segment_col: str,
    target_col: str,
    min_count: int = 10
) -> pd.DataFrame:
    """
    Compute the proportion of each obesity level within groups defined by a lifestyle categorical feature.

    Args:
        df (pd.DataFrame): Input dataframe.
        segment_col (str): Categorical feature for segmentation (e.g., 'FAVC', 'SMOKE', 'MTRANS').
        target_col (str): Name of the target column.
        min_count (int): Minimum group size to report; smaller groups are labeled 'Other'.

    Returns:
        pd.DataFrame: Obesity rate table (rows=segment categories, columns=target classes, values=proportions).

    Raises:
        ValueError: If columns are missing.

    Return schema:
        pd.DataFrame: index = segment_col, columns = sorted(target_col classes), values = proportion (float)

    Domain rationale:
        Lifestyle habits may stratify obesity risk; this table helps identify high-risk subgroups.

    Example:
        >>> lifestyle_segment_obesity_rate(df, 'SMOKE', 'NObeyesdad')
    """
    missing = [c for c in [segment_col, target_col] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    val_counts = df[segment_col].value_counts()
    rare = val_counts[val_counts < min_count].index
    df_segmented = df.copy()
    df_segmented[segment_col] = df_segmented[segment_col].replace(rare, 'Other')
    ctab = pd.crosstab(df_segmented[segment_col], df_segmented[target_col], normalize='index')
    ctab = ctab.loc[:, sorted(ctab.columns)]
    return ctab


# ---------------------------
# Generic/Reusable EDA Tools
# ---------------------------

def summarize_numerical_features(
    df: pd.DataFrame,
    numerical_cols: List[str]
) -> pd.DataFrame:
    """
    Summarize basic statistics for numerical features.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (List[str]): List of numerical columns.

    Returns:
        pd.DataFrame: Table of statistics (mean, std, min, max, median, count, n_unique, n_missing).

    Raises:
        ValueError: If columns are missing.

    Return schema:
        pd.DataFrame with columns:
            - feature (str)
            - mean, std, min, max, median, count, n_unique, n_missing

    Example:
        >>> summarize_numerical_features(df, ['Age', 'BMI'])
    """
    missing = [c for c in numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    summaries = []
    for col in numerical_cols:
        s = df[col]
        summaries.append({
            'feature': col,
            'mean': s.mean(),
            'std': s.std(),
            'min': s.min(),
            'max': s.max(),
            'median': s.median(),
            'count': s.count(),
            'n_unique': s.nunique(),
            'n_missing': s.isnull().sum()
        })
    return pd.DataFrame(summaries)


def summarize_categorical_features(
    df: pd.DataFrame,
    categorical_cols: List[str]
) -> pd.DataFrame:
    """
    Summarize counts, unique values, and missingness for categorical features.

    Args:
        df (pd.DataFrame): Input dataframe.
        categorical_cols (List[str]): List of categorical columns.

    Returns:
        pd.DataFrame: Table with feature, n_unique, top (mode), top_count, n_missing.

    Raises:
        ValueError: If columns are missing.

    Return schema:
        pd.DataFrame with columns:
            - feature (str)
            - n_unique (int)
            - top (object)
            - top_count (int)
            - n_missing (int)

    Example:
        >>> summarize_categorical_features(df, ['Gender', 'FAVC'])
    """
    missing = [c for c in categorical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    summaries = []
    for col in categorical_cols:
        s = df[col]
        top = s.mode().iloc[0] if not s.mode().empty else np.nan
        top_count = s.value_counts().iloc[0] if not s.value_counts().empty else 0
        summaries.append({
            'feature': col,
            'n_unique': s.nunique(),
            'top': top,
            'top_count': top_count,
            'n_missing': s.isnull().sum()
        })
    return pd.DataFrame(summaries)


def automated_outlier_detector(
    df: pd.DataFrame,
    numerical_cols: List[str],
    method: str = 'iqr',
    threshold: float = 1.5
) -> pd.DataFrame:
    """
    Detect outliers in numerical columns using the IQR or Z-score method.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (List[str]): List of numerical columns.
        method (str): 'iqr' or 'zscore'.
        threshold (float): Threshold multiplier for outlier detection.

    Returns:
        pd.DataFrame: Table with feature, n_outliers, outlier_ratio.

    Raises:
        ValueError: If columns are missing or method invalid.

    Return schema:
        pd.DataFrame with columns:
            - feature (str)
            - n_outliers (int)
            - outlier_ratio (float)

    Example:
        >>> automated_outlier_detector(df, ['BMI'], method='iqr', threshold=1.5)
    """
    if method not in {'iqr', 'zscore'}:
        raise ValueError("method must be 'iqr' or 'zscore'")
    missing = [c for c in numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    results = []
    for col in numerical_cols:
        s = df[col].dropna()
        if s.empty:
            results.append({'feature': col, 'n_outliers': 0, 'outlier_ratio': 0.0})
            continue
        if method == 'iqr':
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1
            lower = q1 - threshold * iqr
            upper = q3 + threshold * iqr
            mask = (s < lower) | (s > upper)
        else:  # zscore
            mean = s.mean()
            std = s.std()
            if std == 0:
                mask = pd.Series([False]*len(s), index=s.index)
            else:
                mask = np.abs((s - mean) / std) > threshold
        n_outliers = int(mask.sum())
        outlier_ratio = float(n_outliers) / len(s)
        results.append({'feature': col, 'n_outliers': n_outliers, 'outlier_ratio': outlier_ratio})
    return pd.DataFrame(results)


def plot_numerical_distributions(
    df: pd.DataFrame,
    numerical_cols: List[str],
    n_cols: int = 3,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Create histograms for all numerical features.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_cols (List[str]): List of numerical columns.
        n_cols (int): Number of columns in subplot grid.
        figsize (Tuple[int, int]): Figure size.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        ValueError: If columns are missing.

    Return schema:
        matplotlib.figure.Figure

    Example:
        >>> fig = plot_numerical_distributions(df, ['Age', 'Weight'])
    """
    missing = [c for c in numerical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    n_plots = len(numerical_cols)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for idx, col in enumerate(numerical_cols):
        ax = axes[idx]
        sns.histplot(df[col].dropna(), kde=True, ax=ax, bins=30, color='skyblue')
        ax.set_title(col)
    for ax in axes[n_plots:]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def plot_categorical_counts(
    df: pd.DataFrame,
    categorical_cols: List[str],
    n_cols: int = 3,
    figsize: Tuple[int, int] = (16, 8)
) -> plt.Figure:
    """
    Create countplots for all categorical features.

    Args:
        df (pd.DataFrame): Input dataframe.
        categorical_cols (List[str]): List of categorical columns.
        n_cols (int): Number of columns in subplot grid.
        figsize (Tuple[int, int]): Figure size.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        ValueError: If columns are missing.

    Return schema:
        matplotlib.figure.Figure

    Example:
        >>> fig = plot_categorical_counts(df, ['Gender', 'FAVC'])
    """
    missing = [c for c in categorical_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataframe: {missing}")
    n_plots = len(categorical_cols)
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    for idx, col in enumerate(categorical_cols):
        ax = axes[idx]
        sns.countplot(x=col, data=df, ax=ax, order=sorted(df[col].dropna().unique()))
        ax.set_title(col)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    for ax in axes[n_plots:]:
        ax.axis('off')
    fig.tight_layout()
    return fig


def feature_vs_target_boxplot(
    df: pd.DataFrame,
    numerical_col: str,
    target_col: str,
    figsize: Tuple[int, int] = (8, 5)
) -> plt.Figure:
    """
    Create a boxplot of a numerical feature grouped by target class.

    Args:
        df (pd.DataFrame): Input dataframe.
        numerical_col (str): Numerical column to plot.
        target_col (str): Target/categorical column for grouping.
        figsize (Tuple[int, int]): Figure size.

    Returns:
        matplotlib.figure.Figure: The generated figure.

    Raises:
        ValueError: If columns are missing.

    Return schema:
        matplotlib.figure.Figure

    Example:
        >>> fig = feature_vs_target_boxplot(df, 'BMI', 'NObeyesdad')
    """
    if numerical_col not in df.columns or target_col not in df.columns:
        raise ValueError(f"Missing columns: {[c for c in [numerical_col, target_col] if c not in df.columns]}")
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(x=target_col, y=numerical_col, data=df, ax=ax, order=sorted(df[target_col].dropna().unique()))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_title(f"{numerical_col} by {target_col}")
    return fig


def analyze_target_distribution(
    df: pd.DataFrame,
    target_col: str
) -> pd.DataFrame:
    """
    Summarize counts and proportions of each class in the target variable.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_col (str): Name of the target column.

    Returns:
        pd.DataFrame: Table with class, count, and proportion.

    Raises:
        ValueError: If column is missing.

    Return schema:
        pd.DataFrame with columns:
            - class (object)
            - count (int)
            - proportion (float)

    Example:
        >>> analyze_target_distribution(df, 'NObeyesdad')
    """
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")
    counts = df[target_col].value_counts(dropna=False)
    proportions = counts / counts.sum()
    result = pd.DataFrame({'class': counts.index, 'count': counts.values, 'proportion': proportions.values})
    return result
