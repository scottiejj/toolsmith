## handle_missing_values

**Name:** handle_missing_values  
**Description:** Handle missing values in a specified column of the DataFrame.  
**Applicable Situations:** fill missing values in a specific column

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.
- `column`:
  - **Type:** `string`
  - **Description:** The column name to handle missing values.
- `strategy`:
  - **Type:** `string`
  - **Description:** The strategy to use for imputation.
  - **Enum:** `mean` | `median` | `most_frequent`
  - **Default:** `mean`

**Required:** `df`, `column`  
**Result:** Successfully handle missing values in the specified column  
**Notes:**
- If the column does not exist, a ValueError is raised.
- The strategy must be one of the specified options.
- The imputation method will modify the DataFrame in place.

---

## remove_outliers

**Name:** remove_outliers  
**Description:** Remove outliers from a specified column using the IQR method.  
**Applicable Situations:** clean data by eliminating outliers from a numerical feature

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.
- `column`:
  - **Type:** `string`
  - **Description:** The column name to check for outliers.
- `threshold`:
  - **Type:** `float`
  - **Description:** The multiplier for the IQR to define outliers.
  - **Default:** `1.5`

**Required:** `df`, `column`  
**Result:** DataFrame with outliers removed from the specified column  
**Notes:**
- If the column does not exist or is not numerical, a ValueError is raised.
- Outliers are defined as points beyond the lower and upper bounds calculated from the IQR.

---

## convert_column_to_datetime

**Name:** convert_column_to_datetime  
**Description:** Convert a specified column to datetime format.  
**Applicable Situations:** convert string dates to datetime objects for analysis

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.
- `column`:
  - **Type:** `string`
  - **Description:** The column name to convert.
- `format`:
  - **Type:** `string`
  - **Description:** The format string for datetime conversion.
  - **Default:** `None`

**Required:** `df`, `column`  
**Result:** Successfully convert the specified column to datetime format  
**Notes:**
- If the column does not exist, a ValueError is raised.
- Invalid date formats will be coerced to NaT (Not a Time).

---

## fill_categorical_with_mode

**Name:** fill_categorical_with_mode  
**Description:** Fill missing values in a categorical column with the mode.  
**Applicable Situations:** handle missing values in categorical features

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.
- `column`:
  - **Type:** `string`
  - **Description:** The column name to fill.

**Required:** `df`, `column`  
**Result:** Column with missing values filled with the mode  
**Notes:**
- If the column does not exist or is not categorical, a ValueError is raised.
- The mode is calculated based on the most frequent value in the column.

---

## plot_missing_data_heatmap

**Name:** plot_missing_data_heatmap  
**Description:** Plot a heatmap of missing data in the DataFrame.  
**Applicable Situations:** visualize the distribution of missing values in the dataset

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.

**Required:** `df`  
**Result:** Displays a heatmap of missing values  
**Notes:**
- This function does not return any value; it directly plots the heatmap.
- Ensure that the DataFrame is not empty to avoid plotting errors.

---

## standardize_column

**Name:** standardize_column  
**Description:** Standardize a specified numerical column to have a mean of 0 and standard deviation of 1.  
**Applicable Situations:** prepare numerical features for modeling by standardizing

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.
- `column`:
  - **Type:** `string`
  - **Description:** The column name to standardize.

**Required:** `df`, `column`  
**Result:** The standardized column  
**Notes:**
- If the column does not exist or is not numerical, a ValueError is raised.
- Standardization modifies the DataFrame in place.

---

## encode_categorical

**Name:** encode_categorical  
**Description:** Encode a categorical column into numerical format using one-hot encoding.  
**Applicable Situations:** convert categorical features into a usable format for modeling

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.
- `column`:
  - **Type:** `string`
  - **Description:** The column name to encode.

**Required:** `df`, `column`  
**Result:** DataFrame with the encoded column  
**Notes:**
- If the column does not exist, a ValueError is raised.
- One-hot encoding will drop the first category to avoid multicollinearity.

---

## fix_inconsistent_categories

**Name:** fix_inconsistent_categories  
**Description:** Fix inconsistent categories in a specified categorical column.  
**Applicable Situations:** standardize categorical values to ensure consistency

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The input DataFrame.
- `column`:
  - **Type:** `string`
  - **Description:** The column name to fix.
- `inconsistent_values`:
  - **Type:** `dict`
  - **Description:** A dictionary mapping inconsistent values to the correct value.

**Required:** `df`, `column`, `inconsistent_values`  
**Result:** The column with fixed categories  
**Notes:**
- If the column does not exist, a ValueError is raised.
- Inconsistent values will be replaced in the DataFrame in place.