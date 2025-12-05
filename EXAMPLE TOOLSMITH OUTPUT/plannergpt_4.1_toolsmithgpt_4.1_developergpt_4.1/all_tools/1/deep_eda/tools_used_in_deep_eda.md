## segment_bmi_by_obesity_class

**Name:** segment_bmi_by_obesity_class  
**Description:** Segment BMI statistics by obesity class. Produces summary statistics (mean, std, min, max, median, count) of BMI for each obesity class, optionally excluding BMI outliers.  
**Applicable Situations:** Analyze BMI distribution across target obesity classes, verify class separability or labeling consistency.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe containing BMI, target, and (optionally) outlier flag columns.
- `bmi_col`:
  - **Type:** `string`
  - **Description:** Name of the BMI column.
- `target_col`:
  - **Type:** `string`
  - **Description:** Name of the target column (obesity class).
- `outlier_flag_col`:
  - **Type:** `string | None`
  - **Description:** Optional column indicating BMI outliers; if provided, outliers will be excluded.
  - **Default:** `None`

**Required:** `df`, `bmi_col`, `target_col`  
**Result:** DataFrame with BMI summary statistics for each obesity class  
**Notes:**
- Raises `ValueError` if required columns are missing.
- - Domain rationale: Obesity level is directly related to BMI; this tool quantifies BMI characteristics per class to reveal separation, overlaps, or labeling inconsistencies.



## target_stratified_numerical_summary

**Name:** target_stratified_numerical_summary  
**Description:** Compute summary statistics (mean, std, min, max, median, count) of numerical features stratified by target classes.  
**Applicable Situations:** Explore how health/lifestyle features differ across obesity groups.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `numerical_cols`:
  - **Type:** `array`
  - **Description:** List of numerical columns to summarize.
- `target_col`:
  - **Type:** `string`
  - **Description:** Name of the target column.

**Required:** `df`, `numerical_cols`, `target_col`  
**Result:** DataFrame with multi-column summary statistics for each target class  
**Notes:**
- Raises `ValueError` if any specified columns are missing.
- - Domain rationale: Obesity risk manifests differently across subgroups; stratified summaries reveal discriminative patterns and can guide feature engineering.



## categorical_vs_target_chi2

**Name:** categorical_vs_target_chi2  
**Description:** Perform Chi-squared test between categorical features and the target variable to assess association strength.  
**Applicable Situations:** Identify which categorical features are statistically linked to obesity class.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `cat_cols`:
  - **Type:** `array`
  - **Description:** List of categorical columns.
- `target_col`:
  - **Type:** `string`
  - **Description:** Name of the target column.

**Required:** `df`, `cat_cols`, `target_col`  
**Result:** DataFrame with columns [`feature`, `chi2`, `p_value`, `dof`, `significant`]  
**Notes:**
- Returns NaN for features with insufficient classes.
- Raises `ValueError` if required columns are missing.
- - Domain rationale: Categorical features (e.g. family history, food habits) are hypothesized to be linked to obesity risk. Chi2 tests quantify dependence strength.



## target_vs_feature_anova

**Name:** target_vs_feature_anova  
**Description:** Perform ANOVA (or fallback to Kruskal-Wallis) for each numerical feature across target classes to test for group differences.  
**Applicable Situations:** Quantify whether feature means/medians differ significantly between obesity classes.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `numerical_cols`:
  - **Type:** `array`
  - **Description:** List of numerical columns.
- `target_col`:
  - **Type:** `string`
  - **Description:** Name of the target column.

**Required:** `df`, `numerical_cols`, `target_col`  
**Result:** DataFrame with columns [`feature`, `test`, `stat`, `p_value`, `significant`]  
**Notes:**
- Uses ANOVA if possible, falls back to Kruskal-Wallis for non-normal distributions or errors.
- Returns NaN if groups too small for test.
- Raises `ValueError` if columns are missing.
- - Domain rationale: Quantifies whether means/medians of numerical features differ significantly between obesity classes.



## feature_interaction_heatmap

**Name:** feature_interaction_heatmap  
**Description:** Generate a heatmap of pairwise correlations (Pearson or Spearman) among numerical features.  
**Applicable Situations:** Visualize feature dependencies, multicollinearity, or redundancy.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `numerical_cols`:
  - **Type:** `array`
  - **Description:** List of numerical columns to correlate.
- `method`:
  - **Type:** `string`
  - **Description:** Correlation method.
  - **Enum:** `pearson` | `spearman`
  - **Default:** `pearson`
- `figsize`:
  - **Type:** `tuple`
  - **Description:** Figure size for the heatmap.
  - **Default:** `(10, 8)`

**Required:** `df`, `numerical_cols`  
**Result:** `matplotlib.figure.Figure` heatmap object  
**Notes:**
- Raises `ValueError` if columns missing or method invalid.
- - Domain rationale: Identifies which health/lifestyle features are interdependent, possibly confounded, or redundant for modeling.



## lifestyle_segment_obesity_rate

**Name:** lifestyle_segment_obesity_rate  
**Description:** Compute the proportion of each obesity level within groups defined by a lifestyle categorical feature.  
**Applicable Situations:** Identify high-risk or protective lifestyle subgroups.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `segment_col`:
  - **Type:** `string`
  - **Description:** Categorical feature for segmentation (e.g., 'FAVC', 'SMOKE', 'MTRANS').
- `target_col`:
  - **Type:** `string`
  - **Description:** Name of the target column.
- `min_count`:
  - **Type:** `int`
  - **Description:** Minimum group size to report; smaller groups are labeled 'Other'.
  - **Default:** `10`

**Required:** `df`, `segment_col`, `target_col`  
**Result:** DataFrame with rows=segment categories, columns=obesity classes, values=proportions  
**Notes:**
- Groups smaller than `min_count` are labeled 'Other' to avoid misleading rates.
- Raises `ValueError` if columns are missing.
- - Domain rationale: Lifestyle habits may stratify obesity risk; this table helps identify high-risk subgroups.



## summarize_numerical_features

**Name:** summarize_numerical_features  
**Description:** Summarize basic statistics for numerical features, including mean, std, min, max, median, count, n_unique, n_missing.  
**Applicable Situations:** Get overall sense of numerical data distribution, spot potential anomalies.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `numerical_cols`:
  - **Type:** `array`
  - **Description:** List of numerical columns.

**Required:** `df`, `numerical_cols`  
**Result:** DataFrame of statistics for each feature  
**Notes:**
- Raises `ValueError` if specified columns are missing.



## summarize_categorical_features

**Name:** summarize_categorical_features  
**Description:** Summarize counts, unique values, mode, and missingness for categorical features.  
**Applicable Situations:** Audit categorical feature cardinality and completeness.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `categorical_cols`:
  - **Type:** `array`
  - **Description:** List of categorical columns.

**Required:** `df`, `categorical_cols`  
**Result:** DataFrame with summary info for each feature  
**Notes:**
- Raises `ValueError` if specified columns are missing.



## automated_outlier_detector

**Name:** automated_outlier_detector  
**Description:** Detect outliers in numerical columns using the IQR or Z-score method. Returns count and ratio of outliers per feature.  
**Applicable Situations:** Quickly identify features with potential outlier issues.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `numerical_cols`:
  - **Type:** `array`
  - **Description:** List of numerical columns.
- `method`:
  - **Type:** `string`
  - **Description:** Outlier detection method.
  - **Enum:** `iqr` | `zscore`
  - **Default:** `iqr`
- `threshold`:
  - **Type:** `float`
  - **Description:** Threshold multiplier for outlier detection.
  - **Default:** `1.5`

**Required:** `df`, `numerical_cols`  
**Result:** DataFrame with outlier count and ratio per feature  
**Notes:**
- Raises `ValueError` if columns are missing or method is invalid.



## plot_numerical_distributions

**Name:** plot_numerical_distributions  
**Description:** Create histograms for all numerical features, with density overlay.  
**Applicable Situations:** Visualize distributions and spot skew, multimodality, or outliers.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `numerical_cols`:
  - **Type:** `array`
  - **Description:** List of numerical columns.
- `n_cols`:
  - **Type:** `int`
  - **Description:** Number of subplot columns.
  - **Default:** `3`
- `figsize`:
  - **Type:** `tuple`
  - **Description:** Figure size.
  - **Default:** `(16, 8)`

**Required:** `df`, `numerical_cols`  
**Result:** `matplotlib.figure.Figure` with subplots  
**Notes:**
- Raises `ValueError` if columns are missing.
- Returns a matplotlib Figure; does not display or save.



## plot_categorical_counts

**Name:** plot_categorical_counts  
**Description:** Create countplots for all categorical features in a grid.  
**Applicable Situations:** Visualize commonality and balance of categorical feature values.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `categorical_cols`:
  - **Type:** `array`
  - **Description:** List of categorical columns.
- `n_cols`:
  - **Type:** `int`
  - **Description:** Number of subplot columns.
  - **Default:** `3`
- `figsize`:
  - **Type:** `tuple`
  - **Description:** Figure size.
  - **Default:** `(16, 8)`

**Required:** `df`, `categorical_cols`  
**Result:** `matplotlib.figure.Figure` with subplots  
**Notes:**
- Raises `ValueError` if columns are missing.
- Returns a matplotlib Figure; does not display or save.



## feature_vs_target_boxplot

**Name:** feature_vs_target_boxplot  
**Description:** Create a boxplot of a numerical feature grouped by target class.  
**Applicable Situations:** Visualize central tendency and spread of a feature across obesity groups.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `numerical_col`:
  - **Type:** `string`
  - **Description:** Numerical column to plot.
- `target_col`:
  - **Type:** `string`
  - **Description:** Target/categorical column for grouping.
- `figsize`:
  - **Type:** `tuple`
  - **Description:** Figure size.
  - **Default:** `(8, 5)`

**Required:** `df`, `numerical_col`, `target_col`  
**Result:** `matplotlib.figure.Figure` for boxplot  
**Notes:**
- Raises `ValueError` if columns are missing.
- Returns a matplotlib Figure; does not display or save.



## analyze_target_distribution

**Name:** analyze_target_distribution  
**Description:** Summarize counts and proportions of each class in the target variable.  
**Applicable Situations:** Assess class balance and rarity for classification tasks.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input dataframe.
- `target_col`:
  - **Type:** `string`
  - **Description:** Name of the target column.

**Required:** `df`, `target_col`  
**Result:** DataFrame with class, count, and proportion  
**Notes:**
- Raises `ValueError` if target column is missing.

