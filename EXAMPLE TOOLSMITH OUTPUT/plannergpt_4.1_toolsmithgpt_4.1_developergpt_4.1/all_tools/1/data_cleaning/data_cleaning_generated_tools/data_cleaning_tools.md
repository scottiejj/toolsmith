## bmi_validator_and_flagger

**Name:** bmi_validator_and_flagger  
**Description:** Calculate BMI using height and weight columns, flagging implausible BMI values. Adds two columns: calculated BMI and a flag for suspected erroneous BMI values.  
**Applicable Situations:** Identify and handle implausible BMI records in health/obesity datasets

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `height_col`:
  - **Type:** `string`
  - **Description:** Name of the height column (must be in meters).
- `weight_col`:
  - **Type:** `string`
  - **Description:** Name of the weight column (must be in kilograms).
- `bmi_col`:
  - **Type:** `string`
  - **Description:** Name for the new BMI column.
  - **Default:** `"BMI"`
- `flag_col`:
  - **Type:** `string`
  - **Description:** Name for the new flag column.
  - **Default:** `"BMI_flag"`

**Required:** `df`, `height_col`, `weight_col`  
**Result:** Returns a DataFrame with additional columns `{bmi_col}` (float) and `{flag_col}` (bool)  
**Notes:**
- BMI is calculated as weight / (height^2).
- `{flag_col}` is True if BMI < 10 or BMI > 80, indicating likely data error.
- Raises ValueError if height or weight columns are missing or have non-positive entries.
- Domain rationale: BMI is a key factor in obesity classification; flagging implausible values helps ensure data quality for this medical prediction task.

---

## handle_rare_categories

**Name:** handle_rare_categories  
**Description:** Replace rare categories in a categorical column with a single label (e.g., "Rare") based on a frequency threshold.  
**Applicable Situations:** Consolidate infrequent categories in categorical variables to reduce noise and prevent issues during encoding

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `col`:
  - **Type:** `string`
  - **Description:** Categorical column to process.
- `threshold`:
  - **Type:** `int`
  - **Description:** Categories with count less than or equal to this value are replaced.
  - **Default:** `10`
- `new_category`:
  - **Type:** `string`
  - **Description:** The label to use for rare categories.
  - **Default:** `"Rare"`

**Required:** `df`, `col`  
**Result:** Returns a DataFrame where rare categories in `{col}` are replaced by `{new_category}`  
**Notes:**
- The function is generic and applicable to any categorical variable.
- Use before encoding or modeling to avoid high-cardinality problems.
- Raises ValueError if column is missing.

---

## ordinal_categorical_cleaner

**Name:** ordinal_categorical_cleaner  
**Description:** Standardize ordinal categorical columns, imputing/fixing invalid or missing values and ensuring the correct order.  
**Applicable Situations:** Prepare ordinal categorical features for modeling by enforcing clean values and proper ordering

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `col`:
  - **Type:** `string`
  - **Description:** Ordinal categorical column to clean.
- `valid_order`:
  - **Type:** `array`
  - **Description:** List of valid ordered categories.
- `fill_value`:
  - **Type:** ``string`` | `null`
  - **Description:** Value to use for missing or invalid entries.
  - **Default:** `None`

**Required:** `df`, `col`, `valid_order`  
**Result:** Returns a DataFrame where `{col}` is cast as an ordered categorical with cleaned values  
**Notes:**
- Missing or invalid values are set to `fill_value` if provided.
- Output column will be of pandas CategoricalDtype with specified ordering.
- Raises ValueError if column is missing or `valid_order` is empty.
- Domain rationale: Ordinal features like CAEC and CALC are crucial for risk stratification; enforcing valid, consistent ordering maintains their predictive value.

---

## categorical_binary_cleaner

**Name:** categorical_binary_cleaner  
**Description:** Standardize binary categorical columns (e.g., "yes"/"no"), handling case, typos, and optionally imputing/fixing invalids.  
**Applicable Situations:** Clean binary features with inconsistent representation before encoding or modeling

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `col`:
  - **Type:** `string`
  - **Description:** Column to standardize.
- `valid_values`:
  - **Type:** `(string, string)`
  - **Description:** Acceptable binary values.
  - **Default:** `("yes", "no")`
- `fill_value`:
  - **Type:** ``string`` | `null`
  - **Description:** Value for missing or invalids.
  - **Default:** `None`
- `case_insensitive`:
  - **Type:** `bool`
  - **Description:** If True, matches regardless of case.
  - **Default:** `True`

**Required:** `df`, `col`  
**Result:** Returns a DataFrame with `{col}` standardized to only `valid_values` and `fill_value`  
**Notes:**
- Raises ValueError if column is missing or `valid_values` is not length 2.
- Use to ensure features like "family_history_with_overweight" are consistently coded.
- Domain rationale: Many binary features in this dataset use 'yes'/'no'; standardization avoids modeling errors caused by inconsistent labels.

---

## iqr_outlier_flagger

**Name:** iqr_outlier_flagger  
**Description:** Add a boolean column flagging outliers in a numerical feature using the interquartile range (IQR) method.  
**Applicable Situations:** Detect and optionally remove or address outliers in numerical columns

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `col`:
  - **Type:** `string`
  - **Description:** Numerical column to assess.
- `iqr_mult`:
  - **Type:** `float`
  - **Description:** Outlier threshold multiplier.
  - **Default:** `1.5`
- `flag_col`:
  - **Type:** `string` | `null`
  - **Description:** Name for the flag column.
  - **Default:** `"{col}_outlier_flag"`

**Required:** `df`, `col`  
**Result:** Returns a DataFrame with a `{flag_col}` boolean column indicating outliers  
**Notes:**
- Outliers are values outside [Q1 - iqr_mult*IQR, Q3 + iqr_mult*IQR].
- Raises ValueError for missing columns or non-numeric data.

---

## numerical_type_enforcer

**Name:** numerical_type_enforcer  
**Description:** Convert specified columns to numeric type, optionally coercing errors to NaN and filling invalids with a specified value.  
**Applicable Situations:** Ensure data type consistency and handle invalid values in numerical features

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `cols`:
  - **Type:** `array`
  - **Description:** List of column names to convert.
- `coerce`:
  - **Type:** `bool`
  - **Description:** Whether to coerce errors to NaN.
  - **Default:** `True`
- `fill_invalid_with`:
  - **Type:** `number` | `null`
  - **Description:** Value to fill NaN after conversion.
  - **Default:** `None`

**Required:** `df`, `cols`  
**Result:** Returns a DataFrame where specified columns are numeric (float)  
**Notes:**
- Use to clean up columns before modeling or further analysis.
- Raises ValueError if any column is missing.

---

## gender_cleaner

**Name:** gender_cleaner  
**Description:** Standardize the Gender column to valid values (e.g., "Male"/"Female"), correcting case and typos, and handling invalids.  
**Applicable Situations:** Clean the Gender feature for demographic modeling or stratification

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `col`:
  - **Type:** `string`
  - **Description:** Gender column name.
  - **Default:** `"Gender"`
- `valid_values`:
  - **Type:** `(string, string)`
  - **Description:** Acceptable gender values.
  - **Default:** `("Male", "Female")`
- `fill_value`:
  - **Type:** `string` | `null`
  - **Description:** Value for missing or invalids.
  - **Default:** `None`

**Required:** `df`, `col`  
**Result:** Returns a DataFrame with `{col}` standardized to only valid values and `fill_value`  
**Notes:**
- Raises ValueError if column is missing or `valid_values` is not length 2.
- Domain rationale: Gender is widely used in risk modeling for obesity; ensuring clean, valid entries supports demographic analysis and fairness.

---

## drop_uninformative_id_column

**Name:** drop_uninformative_id_column  
**Description:** Remove the ID column from the DataFrame if present (useful before model training, not for submission creation).  
**Applicable Situations:** Remove identifier columns that are not predictive

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `id_col`:
  - **Type:** `string`
  - **Description:** Name of the ID column.
  - **Default:** `"id"`

**Required:** `df`  
**Result:** Returns a DataFrame without the `{id_col}` column if present  
**Notes:**
- No error is raised if the column is not present.
- Use before modeling to prevent leakage from IDs.

---

## impute_missing_by_group_median

**Name:** impute_missing_by_group_median  
**Description:** Impute missing values in a numerical column using the median value within each group of a categorical feature.  
**Applicable Situations:** Impute missing values in a way that preserves group-level patterns

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `target_col`:
  - **Type:** `string`
  - **Description:** Numerical column to impute.
- `group_col`:
  - **Type:** `string`
  - **Description:** Categorical column to group by.

**Required:** `df`, `target_col`, `group_col`  
**Result:** Returns a DataFrame with missing values in `{target_col}` imputed by group median  
**Notes:**
- Useful for features like Weight or Height where group (e.g., Gender) may affect central tendency.
- Raises ValueError if columns are missing.

---

## cap_outliers

**Name:** cap_outliers  
**Description:** Cap (winsorize) values in a numerical column at specified lower and upper quantiles to limit the impact of extreme outliers.  
**Applicable Situations:** Reduce the influence of extreme values in numerical features

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `col`:
  - **Type:** `string`
  - **Description:** Numerical column to cap.
- `lower_quantile`:
  - **Type:** `float`
  - **Description:** Lower quantile to cap at.
  - **Default:** `0.01`
- `upper_quantile`:
  - **Type:** `float`
  - **Description:** Upper quantile to cap at.
  - **Default:** `0.99`

**Required:** `df`, `col`  
**Result:** Returns a DataFrame where `{col}` is capped at the specified quantiles  
**Notes:**
- Raises ValueError if column is missing.
- Use to prevent extreme values from distorting analyses or models.

---