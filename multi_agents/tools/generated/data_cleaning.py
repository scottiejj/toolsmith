import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt

def handle_missing_values(df: pd.DataFrame, column: str, strategy: str = 'mean') -> pd.Series:
    """
    Handle missing values in a specified column of the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to handle missing values.
        strategy (str): The strategy to use for imputation ('mean', 'median', 'most_frequent'). Default is 'mean'.
        
    Returns:
        pd.Series: The column with missing values handled.
        
    Raises:
        ValueError: If the column does not exist in the DataFrame or if an invalid strategy is provided.
        
    Example:
        cleaned_column = handle_missing_values(df, 'Age', strategy='median')
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    if strategy not in ['mean', 'median', 'most_frequent']:
        raise ValueError("Invalid strategy. Choose from 'mean', 'median', or 'most_frequent'.")
        
    imputer = SimpleImputer(strategy=strategy)
    df[column] = imputer.fit_transform(df[[column]])
    return df[column]

def remove_outliers(df: pd.DataFrame, column: str, threshold: float = 1.5) -> pd.DataFrame:
    """
    Remove outliers from a specified column using the IQR method.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to check for outliers.
        threshold (float): The multiplier for the IQR to define outliers. Default is 1.5.
        
    Returns:
        pd.DataFrame: DataFrame with outliers removed.
        
    Raises:
        ValueError: If the column does not exist in the DataFrame or if it is not numerical.
        
    Example:
        cleaned_df = remove_outliers(df, 'Fare', threshold=1.5)
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric to detect outliers.")
    
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def convert_column_to_datetime(df: pd.DataFrame, column: str, format: str = None) -> pd.Series:
    """
    Convert a specified column to datetime format.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to convert.
        format (str): The format string for datetime conversion. Default is None.
        
    Returns:
        pd.Series: The column converted to datetime.
        
    Raises:
        ValueError: If the column does not exist in the DataFrame.
        
    Example:
        df['Date'] = convert_column_to_datetime(df, 'Date', format='%Y-%m-%d')
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    df[column] = pd.to_datetime(df[column], format=format, errors='coerce')
    return df[column]

def fill_categorical_with_mode(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Fill missing values in a categorical column with the mode.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to fill.
        
    Returns:
        pd.Series: The column with missing values filled with the mode.
        
    Raises:
        ValueError: If the column does not exist in the DataFrame or is not categorical.
        
    Example:
        df['Embarked'] = fill_categorical_with_mode(df, 'Embarked')
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    if not pd.api.types.is_categorical_dtype(df[column]) and not pd.api.types.is_object_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be categorical or object type to fill with mode.")
    
    mode = df[column].mode()[0]
    df[column].fillna(mode, inplace=True)
    return df[column]

def plot_missing_data_heatmap(df: pd.DataFrame) -> None:
    """
    Plot a heatmap of missing data in the DataFrame.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        
    Returns:
        None: Displays a heatmap of missing values.
        
    Example:
        plot_missing_data_heatmap(df)
    """
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Data Heatmap')
    plt.show()

def standardize_column(df: pd.DataFrame, column: str) -> pd.Series:
    """
    Standardize a specified numerical column to have a mean of 0 and standard deviation of 1.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to standardize.
        
    Returns:
        pd.Series: The standardized column.
        
    Raises:
        ValueError: If the column does not exist in the DataFrame or is not numerical.
        
    Example:
        df['Fare'] = standardize_column(df, 'Fare')
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric to standardize.")
    
    df[column] = (df[column] - df[column].mean()) / df[column].std()
    return df[column]

def encode_categorical(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """
    Encode a categorical column into numerical format using one-hot encoding.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to encode.
        
    Returns:
        pd.DataFrame: The DataFrame with the encoded column.
        
    Raises:
        ValueError: If the column does not exist in the DataFrame.
        
    Example:
        df_encoded = encode_categorical(df, 'Embarked')
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    df_encoded = pd.get_dummies(df, columns=[column], drop_first=True)
    return df_encoded

def fix_inconsistent_categories(df: pd.DataFrame, column: str, inconsistent_values: dict) -> pd.Series:
    """
    Fix inconsistent categories in a specified categorical column.
    
    Args:
        df (pd.DataFrame): The input DataFrame.
        column (str): The column name to fix.
        inconsistent_values (dict): A dictionary mapping inconsistent values to the correct value.
        
    Returns:
        pd.Series: The column with fixed categories.
        
    Raises:
        ValueError: If the column does not exist in the DataFrame.
        
    Example:
        df['Cabin'] = fix_inconsistent_categories(df, 'Cabin', {'C': 'C1', 'C2': 'C1'})
    """
    if column not in df.columns:
        raise ValueError(f"Column '{column}' does not exist in the DataFrame.")
    
    df[column].replace(inconsistent_values, inplace=True)
    return df[column]
