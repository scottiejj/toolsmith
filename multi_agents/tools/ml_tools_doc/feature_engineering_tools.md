## create_polynomial_features

**Name:** create_polynomial_features  
**Description:** Generate polynomial features from the specified feature columns.  
**Applicable Situations:** when you need to enhance model performance with polynomial feature interactions

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame containing the data.
- `feature_cols`:
  - **Type:** ``string` | `array``
  - **Description:** List of column names to generate polynomial features from.
- `degree`:
  - **Type:** `int`
  - **Description:** The degree of the polynomial features.
  - **Default:** `2`

**Required:** `df`, `feature_cols`  
**Result:** DataFrame with polynomial features added  
**Notes:**
- Feature columns must exist in the DataFrame; otherwise, a ValueError is raised.
- The polynomial features are created without including the bias (constant term).
- The output DataFrame retains the original DataFrame's index.

---

## one_hot_encode

**Name:** one_hot_encode  
**Description:** Perform one-hot encoding on specified categorical columns.  
**Applicable Situations:** when you need to convert categorical variables into a numerical format suitable for machine learning

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame containing the data.
- `categorical_cols`:
  - **Type:** ``string` | `array``
  - **Description:** List of column names to one-hot encode.

**Required:** `df`, `categorical_cols`  
**Result:** DataFrame with one-hot encoded columns added  
**Notes:**
- Categorical columns must exist in the DataFrame; otherwise, a ValueError is raised.
- The first category for each column is dropped to avoid multicollinearity.
- The output DataFrame contains both the original and the newly created one-hot encoded columns.

---

## fill_missing_values

**Name:** fill_missing_values  
**Description:** Fill missing values in numerical columns using specified strategy.  
**Applicable Situations:** when you need to address missing data issues in your dataset

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame containing the data.
- `strategy`:
  - **Type:** `string`
  - **Description:** Strategy to use for filling missing values.
  - **Enum:** `mean` | `median` | `most_frequent`
  - **Default:** `mean`
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** Specific columns to fill. If `None`, fill all numeric columns.
  - **Default:** `None`

**Required:** `df`  
**Result:** DataFrame with missing values filled  
**Notes:**
- If the specified strategy is not recognized, a ValueError is raised.
- If no columns are specified, all numeric columns are filled using the chosen strategy.
- Using inappropriate strategies for categorical columns will raise an error.

---

## zscore_normalization

**Name:** zscore_normalization  
**Description:** Normalize specified columns using Z-score normalization.  
**Applicable Situations:** when you need to standardize features for better model training

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame containing the data.
- `columns`:
  - **Type:** ``string` | `array``
  - **Description:** List of column names to normalize.

**Required:** `df`, `columns`  
**Result:** DataFrame with Z-score normalized columns  
**Notes:**
- Specified columns must exist in the DataFrame; otherwise, a ValueError is raised.
- The Z-score normalization scales features based on their mean and standard deviation.
- Normalizing non-numeric columns will raise an error.

---

## target_encoding

**Name:** target_encoding  
**Description:** Apply target encoding to categorical columns based on the target variable.  
**Applicable Situations:** when you need to encode categorical variables while considering their relationship with the target variable

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame containing the data.
- `target_col`:
  - **Type:** `string`
  - **Description:** The target column name.
- `categorical_cols`:
  - **Type:** ``string` | `array``
  - **Description:** List of categorical column names to encode.
- `alpha`:
  - **Type:** `float`
  - **Description:** Smoothing factor to prevent overfitting.
  - **Default:** `0.1`

**Required:** `df`, `target_col`, `categorical_cols`  
**Result:** DataFrame with target encoded columns  
**Notes:**
- Target column and categorical columns must exist in the DataFrame; otherwise, a ValueError is raised.
- The encoding helps to mitigate overfitting through smoothing.
- Consider the potential leakage of information from the target variable when using this technique.

---

## interaction_features

**Name:** interaction_features  
**Description:** Create interaction features from specified pairs of feature columns.  
**Applicable Situations:** when you want to explore interactions between features to enhance model complexity

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame containing the data.
- `feature_pairs`:
  - **Type:** ``array` | `tuple``
  - **Description:** List of tuples containing pairs of column names.

**Required:** `df`, `feature_pairs`  
**Result:** DataFrame with interaction features added  
**Notes:**
- Each feature pair must exist in the DataFrame; otherwise, a ValueError is raised.
- Interaction features are created by multiplying the specified pairs of columns.
- The resulting features can help capture non-linear relationships in the data.

---