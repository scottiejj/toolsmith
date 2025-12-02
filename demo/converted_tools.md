```markdown
## fill_missing_values

**Name:**  fill_missing_values  
**Description:**  Fill missing values in specified columns of a DataFrame. This tool can handle both numerical and categorical features by using different filling methods.  
**Applicable Situations:**  handle missing values in various types of features

**Parameters:**
- `data`:
  - **Type:**  `pd.DataFrame`
  - **Description:**  A pandas DataFrame object representing the dataset.
- `columns`:
  - **Type:**  ``string` | `array``
  - **Description:**  The name(s) of the column(s) where missing values should be filled.
- `method`:
  - **Type:**  `string`
  - **Description:**  The method to use for filling missing values.
  - **Enum:**  `auto` | `mean` | `median` | `mode` | `constant`
  - **Default:**  `auto`
- `fill_value`:
  - **Type:**  ``number` | `string` | `null``
  - **Description:**  The value to use when method is 'constant'.
  - **Default:**  `None`

**Required:**  `data`, `columns`  
**Result:**  Successfully fill missing values in the specified column(s) of data  
**Notes:**
- The 'auto' method uses mean for numeric columns and mode for non-numeric columns.
- Using 'mean' or 'median' on non-numeric columns will raise an error.
- The 'mode' method uses the most frequent value, which may not always be appropriate.
- Filling missing values can introduce bias, especially if the data is not missing completely at random.
- Consider the impact of filling missing values on your analysis and model performance.

---

## remove_columns_with_missing_data

**Name:**  remove_columns_with_missing_data  
**Description:**  Remove columns containing missing values from a DataFrame based on a threshold.  
**Applicable Situations:**  remove columns with excessive missing data

**Parameters:**
- `data`:
  - **Type:**  `pd.DataFrame`
  - **Description:**  The input DataFrame.
- `thresh`:
  - **Type:**  `number`
  - **Description:**  The minimum proportion of missing values required to drop a column. Should be between 0 and 1.
  - **Default:**  `0.5`
- `columns`:
  - **Type:**  ``string` | `array``
  - **Description:**  Labels of columns to consider.
  - **Default:**  `None`

**Required:**  `data`  
**Result:**  The DataFrame with columns containing excessive missing values removed  
**Notes:**
- The `thresh` parameter must be between 0 and 1.
- If `columns` is specified, only those columns are considered for removal.
- Columns not specified in `columns` are retained in the final DataFrame.

---

## detect_and_handle_outliers_zscore

**Name:**  detect_and_handle_outliers_zscore  
**Description:**  Detect and handle outliers in specified columns using the Z-score method.  
**Applicable Situations:**  handle outliers in numeric data

**Parameters:**
- `data`:
  - **Type:**  `pd.DataFrame`
  - **Description:**  The input DataFrame.
- `columns`:
  - **Type:**  ``string` | `array``
  - **Description:**  The name(s) of the column(s) to check for outliers.
- `threshold`:
  - **Type:**  `number`
  - **Description:**  The Z-score threshold to identify outliers.
  - **Default:**  `3.0`
- `method`:
  - **Type:**  `string`
  - **Description:**  The method to handle outliers.
  - **Enum:**  `clip` | `remove`
  - **Default:**  `clip`

**Required:**  `data`, `columns`  
**Result:**  The DataFrame with outliers handled  
**Notes:**
- Only numeric columns can be processed; non-numeric columns will raise an error.
- The 'clip' method adjusts outliers to the threshold bounds, while 'remove' deletes them.
- Consider the impact of outlier handling on data distribution and analysis results.

---
```