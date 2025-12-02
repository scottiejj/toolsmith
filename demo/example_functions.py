
def fill_missing_values(data: pd.DataFrame, columns: Union[str, List[str]], method: str = 'auto', fill_value: Any = None) -> pd.DataFrame:
    """
    Fill missing values in specified columns of a DataFrame.

    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): The name(s) of the column(s) to fill missing values.
        method (str, optional): The method to use for filling missing values. 
            Options: 'auto', 'mean', 'median', 'mode', 'constant'. Defaults to 'auto'.
        fill_value (Any, optional): The value to use when method is 'constant'. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with filled missing values.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if method == 'auto':
            if pd.api.types.is_numeric_dtype(data[column]):
                data[column].fillna(data[column].mean(), inplace=True)
            else:
                data[column].fillna(data[column].mode()[0], inplace=True)
        elif method == 'mean':
            data[column].fillna(data[column].mean(), inplace=True)
        elif method == 'median':
            data[column].fillna(data[column].median(), inplace=True)
        elif method == 'mode':
            data[column].fillna(data[column].mode()[0], inplace=True)
        elif method == 'constant':
            data[column].fillna(fill_value, inplace=True)
        else:
            raise ValueError("Invalid method. Choose from 'auto', 'mean', 'median', 'mode', or 'constant'.")

    return data

def remove_columns_with_missing_data(data: pd.DataFrame, thresh: float = 0.5, columns: Union[str, List[str]] = None) -> pd.DataFrame:
    """
    Remove columns containing missing values from a DataFrame based on a threshold.

    Args:
        data (pd.DataFrame): The input DataFrame.
        thresh (float, optional): The minimum proportion of missing values required to drop a column. 
                                    Should be between 0 and 1. Defaults to 0.5.
        columns (str or List[str], optional): Labels of columns to consider. Defaults to None.

    Returns:
        pd.DataFrame: The DataFrame with columns containing excessive missing values removed.
    """
    if not 0 <= thresh <= 1:
        raise ValueError("thresh must be between 0 and 1")

    if columns is not None:
        if isinstance(columns, str):
            columns = [columns]
        data_subset = data[columns]
    else:
        data_subset = data

    # Calculate the number of missing values allowed based on the threshold
    max_missing = int(thresh * len(data_subset))

    # Identify columns to keep
    columns_to_keep = data_subset.columns[data_subset.isna().sum() < max_missing]

    # If columns was specified, add back other columns not in the subset
    if columns is not None:
        columns_to_keep = columns_to_keep.union(data.columns.difference(columns))

    return data[columns_to_keep]

def detect_and_handle_outliers_zscore(data: pd.DataFrame, columns: Union[str, List[str]], threshold: float = 3.0, method: str = 'clip') -> pd.DataFrame:
    """
    Detect and handle outliers in specified columns using the Z-score method.
    Args:
        data (pd.DataFrame): The input DataFrame.
        columns (str or List[str]): The name(s) of the column(s) to check for outliers.
        threshold (float, optional): The Z-score threshold to identify outliers. Defaults to 3.0.
        method (str, optional): The method to handle outliers. Options: 'clip', 'remove'. Defaults to 'clip'.
    Returns:
        pd.DataFrame: The DataFrame with outliers handled.
    """
    if isinstance(columns, str):
        columns = [columns]

    for column in columns:
        if not pd.api.types.is_numeric_dtype(data[column]):
            raise ValueError(f"Column '{column}' must be numeric.")

        mean = data[column].mean()
        std = data[column].std()
        z_scores = (data[column] - mean) / std

        if method == 'clip':
            # Define the bounds
            lower_bound = mean - threshold * std
            upper_bound = mean + threshold * std
            print(lower_bound, upper_bound)
            # Apply clipping only to values exceeding the threshold
            data.loc[z_scores > threshold, column] = upper_bound
            data.loc[z_scores < -threshold, column] = lower_bound
        elif method == 'remove':
            data = data[abs(z_scores) <= threshold]
        else:
            raise ValueError("Invalid method. Choose from 'clip' or 'remove'.")

    return data