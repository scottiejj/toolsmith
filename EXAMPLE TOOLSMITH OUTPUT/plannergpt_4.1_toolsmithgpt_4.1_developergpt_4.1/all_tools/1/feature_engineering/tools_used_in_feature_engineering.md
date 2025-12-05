## create_bmi_category_feature

**Name:** create_bmi_category_feature  
**Description:** Create a BMI category feature based on WHO BMI thresholds.  
**Applicable Situations:** Feature engineering for obesity and health-related datasets; transforming continuous BMI into categorical risk classes.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `bmi_col`:
  - **Type:** `string`
  - **Description:** Column name for BMI.
  - **Default:** `"BMI"`
- `new_col`:
  - **Type:** `string`
  - **Description:** Name for the new BMI category column.
  - **Default:** `"BMI_category"`

**Required:** `df`  
**Result:** DataFrame with new BMI category column  
**Notes:**
- Dataset-specific tool.
- Domain rationale: Encapsulates medical BMI thresholds, making it easier to combine with other features or model non-linear relationships.
- BMI categories:
    - Underweight: < 18.5
    - Normal weight: 18.5 - 24.9
    - Overweight: 25 - 29.9
    - Obesity I: 30 - 34.9
    - Obesity II: 35 - 39.9
    - Obesity III: >= 40



## create_lifestyle_risk_score

**Name:** create_lifestyle_risk_score  
**Description:** Create a composite lifestyle risk score based on dietary and behavioral factors.  
**Applicable Situations:** Summing multiple health-related behaviors into a single numerical risk score for predictive modeling.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `favc_col`:
  - **Type:** `string`
  - **Description:** Column for high-caloric food consumption (yes/no).
  - **Default:** `"FAVC"`
- `caec_col`:
  - **Type:** `string`
  - **Description:** Column for food between meals.
  - **Default:** `"CAEC"`
- `calc_col`:
  - **Type:** `string`
  - **Description:** Column for alcohol consumption.
  - **Default:** `"CALC"`
- `scc_col`:
  - **Type:** `string`
  - **Description:** Column for caloric monitoring (yes/no).
  - **Default:** `"SCC"`
- `smoke_col`:
  - **Type:** `string`
  - **Description:** Column for smoking (yes/no).
  - **Default:** `"SMOKE"`
- `fcvc_col`:
  - **Type:** `string`
  - **Description:** Frequency of vegetable consumption (higher is better).
  - **Default:** `"FCVC"`
- `ch2o_col`:
  - **Type:** `string`
  - **Description:** Water intake (higher is better).
  - **Default:** `"CH2O"`
- `faf_col`:
  - **Type:** `string`
  - **Description:** Physical activity frequency (higher is better).
  - **Default:** `"FAF"`
- `tue_col`:
  - **Type:** `string`
  - **Description:** Time using technology (higher is worse).
  - **Default:** `"TUE"`
- `new_col`:
  - **Type:** `string`
  - **Description:** Name for the new lifestyle risk score column.
  - **Default:** `"lifestyle_risk_score"`

**Required:** `df`  
**Result:** DataFrame with the new lifestyle risk score column  
**Notes:**
- Dataset-specific tool.
- Domain rationale: Obesity is influenced by combined lifestyle factors; this score summarizes risk-promoting behaviors for modeling.
- The score is a weighted sum; higher values indicate higher risk.
- The score includes both negative (risk) and positive (protective) behaviors, with appropriate scaling.



## create_age_lifestyle_interaction

**Name:** create_age_lifestyle_interaction  
**Description:** Create an interaction feature between age and composite lifestyle factors.  
**Applicable Situations:** When age is hypothesized to moderate the impact of lifestyle behaviors on obesity risk.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `age_col`:
  - **Type:** `string`
  - **Description:** Column for age.
  - **Default:** `"Age"`
- `favc_col`:
  - **Type:** `string`
  - **Description:** High-caloric food consumption (yes/no).
  - **Default:** `"FAVC"`
- `fcvc_col`:
  - **Type:** `string`
  - **Description:** Vegetable consumption frequency.
  - **Default:** `"FCVC"`
- `faf_col`:
  - **Type:** `string`
  - **Description:** Physical activity frequency.
  - **Default:** `"FAF"`
- `new_col`:
  - **Type:** `string`
  - **Description:** Name for the new interaction column.
  - **Default:** `"age_lifestyle_interaction"`

**Required:** `df`  
**Result:** DataFrame with the new interaction feature  
**Notes:**
- Dataset-specific tool.
- Domain rationale: Age moderates the impact of lifestyle; this feature captures non-linear risk growth with unhealthy behaviors as age increases.
- Higher values represent older individuals with unhealthy behavior combinations.



## create_bmi_weight_interaction

**Name:** create_bmi_weight_interaction  
**Description:** Create an interaction feature between BMI and Weight.  
**Applicable Situations:** When high correlation between features suggests an interaction term may improve model flexibility or flag redundancy.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `bmi_col`:
  - **Type:** `string`
  - **Description:** BMI column.
  - **Default:** `"BMI"`
- `weight_col`:
  - **Type:** `string`
  - **Description:** Weight column.
  - **Default:** `"Weight"`
- `new_col`:
  - **Type:** `string`
  - **Description:** Name for the new interaction column.
  - **Default:** `"bmi_weight_interaction"`

**Required:** `df`  
**Result:** DataFrame with new interaction column  
**Notes:**
- Dataset-specific tool.
- Domain rationale: High collinearity between BMI and Weight; this term may help models capture non-linear relationships or flag redundant information.



## create_lifestyle_balance_feature

**Name:** create_lifestyle_balance_feature  
**Description:** Create a feature representing the balance between physical activity and sedentary behavior.  
**Applicable Situations:** When predicting outcomes influenced by physical activity and sedentary time.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `faf_col`:
  - **Type:** `string`
  - **Description:** Physical activity frequency.
  - **Default:** `"FAF"`
- `tue_col`:
  - **Type:** `string`
  - **Description:** Time using technology devices.
  - **Default:** `"TUE"`
- `new_col`:
  - **Type:** `string`
  - **Description:** Name for new balance column.
  - **Default:** `"lifestyle_balance"`

**Required:** `df`  
**Result:** DataFrame with new lifestyle balance column  
**Notes:**
- Dataset-specific tool.
- Domain rationale: Obesity risk is affected by the balance of activity (FAF) and sedentary (TUE) time.
- Positive values indicate more physical activity relative to sedentary behavior.



## one_hot_encode_columns

**Name:** one_hot_encode_columns  
**Description:** One-hot encode specified categorical columns.  
**Applicable Situations:** Preparing categorical features for use in machine learning models that require numeric input.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `categorical_cols`:
  - **Type:** `array`
  - **Description:** List of categorical columns to encode.
- `drop_original`:
  - **Type:** `bool`
  - **Description:** If True, drop the original categorical columns.
  - **Default:** `False`
- `prefix_sep`:
  - **Type:** `string`
  - **Description:** Separator for new column names.
  - **Default:** `"_"`

**Required:** `df`, `categorical_cols`  
**Result:** DataFrame with one-hot encoded columns (and optionally originals dropped)  
**Notes:**
- Generic tool.
- All unique categories present in the data are encoded.



## target_mean_encoding

**Name:** target_mean_encoding  
**Description:** Perform target mean encoding for categorical columns (suitable for train only).  
**Applicable Situations:** Encoding high-cardinality categorical variables for regression or classification using the target's mean.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `categorical_cols`:
  - **Type:** `array`
  - **Description:** Categorical columns to encode.
- `target_col`:
  - **Type:** `string`
  - **Description:** Target variable (must be numeric or ordinal encoded).
- `min_samples_leaf`:
  - **Type:** `int`
  - **Description:** Minimum samples to allow for regularization.
  - **Default:** `1`
- `smoothing`:
  - **Type:** `float`
  - **Description:** Smoothing strength (higher = more global mean).
  - **Default:** `1.0`
- `suffix`:
  - **Type:** `string`
  - **Description:** Suffix for new encoded columns.
  - **Default:** `"_target_enc"`

**Required:** `df`, `categorical_cols`, `target_col`  
**Result:** DataFrame with original columns plus target-encoded columns  
**Notes:**
- Generic tool, but must be applied with care to avoid target leakage (do not fit on full train+test).
- For multiclass targets, must provide ordinal/numeric encoding.
- Fit on train only; apply mapping to test.



## standardize_numerical_features

**Name:** standardize_numerical_features  
**Description:** Standardize numerical features using z-score.  
**Applicable Situations:** When features need to be on a comparable scale for modeling, distance-based algorithms, or PCA.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `numerical_cols`:
  - **Type:** `array`
  - **Description:** List of numerical columns to standardize.
- `suffix`:
  - **Type:** `string`
  - **Description:** Suffix to append to standardized columns.
  - **Default:** `"_std"`

**Required:** `df`, `numerical_cols`  
**Result:** DataFrame with original columns plus standardized versions  
**Notes:**
- Generic tool.
- Missing values are ignored in standardization; standardized columns will contain NaN where data was missing.



## aggregate_flag_features

**Name:** aggregate_flag_features  
**Description:** Aggregate multiple boolean flag columns into a single count feature.  
**Applicable Situations:** Summarizing the number of flags/outliers/anomaly indicators per row.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `flag_cols`:
  - **Type:** `array`
  - **Description:** List of boolean flag columns.
- `new_col`:
  - **Type:** `string`
  - **Description:** Name for the aggregated count column.
  - **Default:** `"n_flags"`

**Required:** `df`, `flag_cols`  
**Result:** DataFrame with original columns plus aggregated flag count  
**Notes:**
- Generic tool.
- Summing over boolean columns; result is integer count per row.

