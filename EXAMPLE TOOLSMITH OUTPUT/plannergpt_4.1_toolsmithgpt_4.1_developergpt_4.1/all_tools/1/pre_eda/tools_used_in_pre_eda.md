## summarize_categorical_features

**Name:** summarize_categorical_features  
**Description:** Summarizes categorical columns: unique values, value counts, and missing values.  
**Applicable Situations:** Get an overview of categorical distributions, class balance, and missing data in categorical features.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The data frame containing the data.
- `categorical_cols`:
  - **Type:** `List[str]`
  - **Description:** List of column names to summarize.

**Required:** `df`, `categorical_cols`  
**Result:** Summary table with columns: 'feature', 'num_unique', 'most_common', 'most_common_freq', 'missing_count'  
**Notes:**
- Raises ValueError if any specified categorical columns are missing.
- Useful before encoding or further processing categorical variables.



## summarize_numerical_features

**Name:** summarize_numerical_features  
**Description:** Provides descriptive statistics for numerical columns.  
**Applicable Situations:** Quickly assess range, central tendency, spread, and missingness in numerical data.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** The data frame containing the data.
- `numerical_cols`:
  - **Type:** `List[str]`
  - **Description:** List of numerical column names.

**Required:** `df`, `numerical_cols`  
**Result:** Descriptive statistics table with columns: 'feature', 'mean', 'std', 'min', '25%', '50%', '75%', 'max', 'missing_count'  
**Notes:**
- Raises ValueError if any specified numerical columns are missing.
- Use this to detect likely outliers or data entry errors.



## plot_numerical_distributions

**Name:** plot_numerical_distributions  
**Description:** Plots histograms (optionally colored by target class) for multiple numerical features.  
**Applicable Situations:** Visualize the distribution of multiple numeric features, optionally stratified by a categorical target.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** DataFrame with data.
- `numerical_cols`:
  - **Type:** `List[str]`
  - **Description:** List of numerical features to plot.
- `target_col`:
  - **Type:** `Optional[str]`
  - **Description:** If provided, use as hue for color separation.
  - **Default:** `None`
- `n_cols`:
  - **Type:** `int`
  - **Description:** Number of columns in subplot grid.
  - **Default:** `3`
- `figsize`:
  - **Type:** `Tuple[int, int]`
  - **Description:** Figure size (width, height).
  - **Default:** `(16, 4)`

**Required:** `df`, `numerical_cols`  
**Result:** The matplotlib Figure object (not shown).  
**Notes:**
- Raises ValueError if any numerical columns are missing.
- Returns the figure object for further processing or saving.



## plot_categorical_counts

**Name:** plot_categorical_counts  
**Description:** Plots countplots for multiple categorical features (optionally colored by target class).  
**Applicable Situations:** Visualize the frequency of categories, optionally by class, to check for imbalance or patterns.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** DataFrame with data.
- `categorical_cols`:
  - **Type:** `List[str]`
  - **Description:** Categorical features to plot.
- `target_col`:
  - **Type:** `Optional[str]`
  - **Description:** Target column for hue grouping.
  - **Default:** `None`
- `n_cols`:
  - **Type:** `int`
  - **Description:** Number of columns in subplot grid.
  - **Default:** `3`
- `figsize`:
  - **Type:** `Tuple[int, int]`
  - **Description:** Figure size (width, height).
  - **Default:** `(16, 4)`

**Required:** `df`, `categorical_cols`  
**Result:** The matplotlib Figure object (not shown).  
**Notes:**
- Raises ValueError if any categorical columns are missing.
- Returns the figure object for further processing or saving.



## analyze_target_distribution

**Name:** analyze_target_distribution  
**Description:** Analyzes the class distribution of the target variable.  
**Applicable Situations:** Assess class balance/imbalance in the target variable for impact on modeling and metric choice.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** DataFrame with target column.
- `target_col`:
  - **Type:** `str`
  - **Description:** The target column name.

**Required:** `df`, `target_col`  
**Result:** Table with 'class', 'count', 'proportion'.  
**Notes:**
- Raises ValueError if the target column is missing.
- - Domain rationale: The dataset's target is a multi-class categorical variable ("NObeyesdad") with potentially imbalanced classes. Understanding class distribution is crucial for modeling and evaluation.



## correlation_matrix_heatmap

**Name:** correlation_matrix_heatmap  
**Description:** Plots a correlation matrix heatmap for numerical features.  
**Applicable Situations:** Visualize linear (or rank-based) relationships among numerical features to detect multicollinearity or clusters.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** DataFrame.
- `numerical_cols`:
  - **Type:** `List[str]`
  - **Description:** List of numerical columns to include.
- `method`:
  - **Type:** `str`
  - **Description:** Correlation method.
  - **Enum:** `pearson` | `spearman` | `kendall`
  - **Default:** `pearson`
- `figsize`:
  - **Type:** `Tuple[int, int]`
  - **Description:** Figure size (width, height).
  - **Default:** `(10, 8)`

**Required:** `df`, `numerical_cols`  
**Result:** The matplotlib Figure object (not shown).  
**Notes:**
- Raises ValueError if any numerical columns are missing or method is invalid.
- Returns the figure object for further processing or saving.



## bmi_feature_analysis

**Name:** bmi_feature_analysis  
**Description:** Computes BMI (Body Mass Index) for each individual, summarizes its distribution, and optionally visualizes by target class.  
**Applicable Situations:** Directly assess a core medical risk factor (BMI) distribution and its relation to obesity classes.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** Input DataFrame.
- `height_col`:
  - **Type:** `str`
  - **Description:** Column for height (meters).
- `weight_col`:
  - **Type:** `str`
  - **Description:** Column for weight (kg).
- `target_col`:
  - **Type:** `Optional[str]`
  - **Description:** If provided, produces a plot colored by this column.
  - **Default:** `None`

**Required:** `df`, `height_col`, `weight_col`  
**Result:** Dict with keys:
  - 'bmi_series': pd.Series (BMI values, indexed as df)
  - 'bmi_summary': pd.Series (basic statistics)
  - 'bmi_plot': plt.Figure (if target_col provided, else None)  
**Notes:**
- Raises ValueError if columns are missing or contain invalid (<=0) values.
- - Domain rationale: BMI is a primary indicator for obesity risk, directly related to the prediction target. Assessing its distribution and its relation to "NObeyesdad" provides critical insights.



## automated_outlier_detector

**Name:** automated_outlier_detector  
**Description:** Identifies outliers in numerical columns using IQR or z-score method.  
**Applicable Situations:** Automated detection and quantification of potential anomalous values in continuous data.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** DataFrame to check.
- `numerical_cols`:
  - **Type:** `List[str]`
  - **Description:** List of numerical columns to check.
- `method`:
  - **Type:** `str`
  - **Description:** Outlier detection method.
  - **Enum:** `iqr` | `zscore`
  - **Default:** `iqr`
- `threshold`:
  - **Type:** `float`
  - **Description:** Outlier threshold (IQR multiple or z-score).
  - **Default:** `1.5`

**Required:** `df`, `numerical_cols`  
**Result:** Table with columns: 'feature', 'outlier_count', 'outlier_ratio', 'method', 'threshold'.  
**Notes:**
- Raises ValueError if any columns are missing, or method invalid.
- Use high thresholds for robust detection, or lower to flag more points.



## target_feature_impact_analysis

**Name:** target_feature_impact_analysis  
**Description:** Quantifies the association between features and the target. For numerical features: computes ANOVA F-statistic and p-value. For categorical features: computes Cramér's V.  
**Applicable Situations:** Identify which features (numerical or categorical) are most associated with the multi-class target for feature selection or interpretation.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** DataFrame.
- `feature_cols`:
  - **Type:** `List[str]`
  - **Description:** Features to analyze.
- `target_col`:
  - **Type:** `str`
  - **Description:** The target variable (must be categorical).
- `feature_types`:
  - **Type:** `Optional[Dict[str, str]]`
  - **Description:** Dict mapping feature name to 'categorical' or 'numerical'. If None, types are inferred.
  - **Default:** `None`

**Required:** `df`, `feature_cols`, `target_col`  
**Result:** Table with columns: 'feature', 'feature_type', 'association', 'p_value' (for numerical), 'method'  
**Notes:**
- Raises ValueError if columns are missing.
- - Domain rationale: The target is multi-class categorical, so understanding which features (diet, lifestyle, etc.) show the strongest association with obesity risk is critical for both feature selection and interpretation.



## feature_vs_target_boxplot

**Name:** feature_vs_target_boxplot  
**Description:** Boxplot of a numerical feature grouped by target classes.  
**Applicable Situations:** Visualize spread and central tendency of a numeric feature across obesity classes.

**Parameters:**
- `df`:
  - **Type:** `pd.DataFrame`
  - **Description:** DataFrame.
- `numerical_col`:
  - **Type:** `str`
  - **Description:** Feature to plot.
- `target_col`:
  - **Type:** `str`
  - **Description:** Target variable.
- `figsize`:
  - **Type:** `Tuple[int, int]`
  - **Description:** Figure size (width, height).
  - **Default:** `(8, 5)`

**Required:** `df`, `numerical_col`, `target_col`  
**Result:** The matplotlib Figure object.  
**Notes:**
- Raises ValueError if columns are missing.
- - Domain rationale: For this dataset, visualizing how BMI, Age, Weight, or other numerics vary by obesity class can reveal separable patterns and inform feature engineering.

