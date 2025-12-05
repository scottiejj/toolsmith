import sys
import os
import importlib.util
import sys as _sys

# Basic paths
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents','multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

phase_module_path = os.path.join('multi_agents', 'competition', "obesity_risks", "deep_eda", "deep_eda_generated_tools", "deep_eda.py")

spec = importlib.util.spec_from_file_location('phase_tools', phase_module_path)
phase_tools = importlib.util.module_from_spec(spec)
_sys.modules['phase_tools'] = phase_tools
spec.loader.exec_module(phase_tools)
from phase_tools import *  

def generated_code_function():
    import numpy as np
    import pandas as pd
phase_module_path = os.path.join('multi_agents', 'competition', "obesity_risks", "data_cleaning", "data_cleaning_generated_tools", "data_cleaning.py")

spec = importlib.util.spec_from_file_location('phase_tools', phase_module_path)
phase_tools = importlib.util.module_from_spec(spec)
_sys.modules['phase_tools'] = phase_tools
spec.loader.exec_module(phase_tools)
from phase_tools import *  

def generated_code_function():
    import numpy as np
    import pandas as pd
    
    import os
    import pandas as pd
    
    # File paths
    data_dir = '/Users/scottiejj/Desktop/AutoKaggle_APAPTED/multi_agents/competition/obesity_risks/'
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    
    # Load data
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    # Work on copies
    train_clean = train.copy()
    test_clean = test.copy()
    
    # 1. Numerical columns
    num_cols = ['Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']
    
    # Fill invalids with train median (compute median for each col in train, apply to both)
    num_medians = {col: train_clean[col].median() for col in num_cols}
    
    for col in num_cols:
        train_clean = numerical_type_enforcer(train_clean, cols=[col], coerce=True, fill_invalid_with=num_medians[col])
        test_clean = numerical_type_enforcer(test_clean, cols=[col], coerce=True, fill_invalid_with=num_medians[col])
    
    # 2. Gender
    train_gender_mode = train_clean['Gender'].mode()[0] if train_clean['Gender'].mode().size > 0 else "Female"
    train_clean = gender_cleaner(train_clean, col='Gender', valid_values=("Male", "Female"), fill_value=train_gender_mode)
    test_clean = gender_cleaner(test_clean, col='Gender', valid_values=("Male", "Female"), fill_value=train_gender_mode)
    
    # 3. Binary Categorical Features
    binary_cols = ['family_history_with_overweight', 'FAVC', 'SMOKE', 'SCC']
    for col in binary_cols:
        mode = train_clean[col].mode()[0] if train_clean[col].mode().size > 0 else "no"
        train_clean = categorical_binary_cleaner(train_clean, col=col, valid_values=("yes", "no"), fill_value=mode)
        test_clean = categorical_binary_cleaner(test_clean, col=col, valid_values=("yes", "no"), fill_value=mode)
    
    # 4. Ordinal Categorical Features
    ordinal_cols = ['CAEC', 'CALC']
    ordinal_order = ["no", "Sometimes", "Frequently", "Always"]
    for col in ordinal_cols:
        train_mode = train_clean[col].mode()[0] if train_clean[col].mode().size > 0 else "no"
        train_clean = ordinal_categorical_cleaner(train_clean, col=col, valid_order=ordinal_order, fill_value=train_mode)
        test_clean = ordinal_categorical_cleaner(test_clean, col=col, valid_order=ordinal_order, fill_value=train_mode)
    
    # 5. Manual Harmonization: MTRANS
    # Normalize string values (strip, lower, replace spaces/underscores for harmonization)
    def harmonize_mtrans(series):
        return (
            series.astype(str)
            .str.strip()
            .str.replace(' ', '_')
            .str.replace('/', '_')
            .str.replace('-', '_')
            .str.replace('__', '_')
            .str.lower()
            .str.capitalize()
        )
    
    train_clean['MTRANS'] = harmonize_mtrans(train_clean['MTRANS'])
    test_clean['MTRANS'] = harmonize_mtrans(test_clean['MTRANS'])
    
    # Ensure all test categories are present in train, otherwise relabel as 'Other'
    valid_mtrans = set(train_clean['MTRANS'].unique())
    test_clean['MTRANS'] = test_clean['MTRANS'].apply(lambda x: x if x in valid_mtrans else 'Other')
    train_clean['MTRANS'] = train_clean['MTRANS'].apply(lambda x: x if x in valid_mtrans else 'Other')
    
    
    # Numerical imputation by group for relevant columns
    group_impute_cols = ['Height', 'Weight', 'Age']
    for col in group_impute_cols:
        train_clean = impute_missing_by_group_median(train_clean, target_col=col, group_col='Gender')
        test_clean = impute_missing_by_group_median(test_clean, target_col=col, group_col='Gender')
    
    # Remaining numerical columns: fill any leftover missing with overall train median (already handled with numerical_type_enforcer in practice)
    for col in ['FCVC', 'NCP', 'CH2O', 'FAF', 'TUE']:
        # If any missing remains (shouldn't, but check just in case)
        if train_clean[col].isnull().any():
            train_clean[col] = train_clean[col].fillna(num_medians[col])
        if test_clean[col].isnull().any():
            test_clean[col] = test_clean[col].fillna(num_medians[col])
    
    
    # 1. Add BMI and BMI_flag
    train_clean = bmi_validator_and_flagger(train_clean, height_col='Height', weight_col='Weight', bmi_col='BMI', flag_col='BMI_flag')
    test_clean = bmi_validator_and_flagger(test_clean, height_col='Height', weight_col='Weight', bmi_col='BMI', flag_col='BMI_flag')
    
    # 2. Handle implausible BMI
    # If BMI < 10 or BMI > 80: in train, drop if far outside; in test, cap at 1st/99th percentile
    # We'll drop train rows with BMI_flag==True and BMI < 8 or BMI > 90 (very extreme), otherwise cap
    
    far_low, far_high = 8, 90
    # For train
    extreme_bmi = train_clean[(train_clean['BMI_flag']) & ((train_clean['BMI'] < far_low) | (train_clean['BMI'] > far_high))].index
    train_clean = train_clean.drop(extreme_bmi)
    # For remaining flagged in train and all flagged in test: cap BMI, Height, Weight at 1st/99th percentile
    
    for col in ['BMI', 'Height', 'Weight']:
        # Compute percentiles from train (excluding dropped rows)
        lower = train_clean[col].quantile(0.01)
        upper = train_clean[col].quantile(0.99)
        train_clean = cap_outliers(train_clean, col=col, lower_quantile=0.01, upper_quantile=0.99)
        test_clean = cap_outliers(test_clean, col=col, lower_quantile=0.01, upper_quantile=0.99)
    
    # 3. IQR outlier flagging and capping for all numerical features (incl. BMI)
    iqr_num_cols = num_cols + ['BMI']
    for col in iqr_num_cols:
        train_clean = iqr_outlier_flagger(train_clean, col=col, iqr_mult=1.5, flag_col=f'{col}_outlier_flag')
        test_clean = iqr_outlier_flagger(test_clean, col=col, iqr_mult=1.5, flag_col=f'{col}_outlier_flag')
        train_clean = cap_outliers(train_clean, col=col, lower_quantile=0.01, upper_quantile=0.99)
        test_clean = cap_outliers(test_clean, col=col, lower_quantile=0.01, upper_quantile=0.99)
    
    
    # Define categorical columns for rare handling
    cat_cols = ['MTRANS', 'CAEC', 'CALC']
    
    for col in cat_cols:
        train_clean = handle_rare_categories(train_clean, col=col, threshold=10, new_category='Rare')
        test_clean = handle_rare_categories(test_clean, col=col, threshold=10, new_category='Rare')
        # Harmonize test categories: if not in train, set to 'Rare'
        valid_cats = set(train_clean[col].unique())
        test_clean[col] = test_clean[col].apply(lambda x: x if x in valid_cats else 'Rare')
        train_clean[col] = train_clean[col].apply(lambda x: x if x in valid_cats else 'Rare')
    
    # Ensure all categorical columns are consistent in spelling/capitalization
    # For all categorical columns: strip, capitalize (except binary, which should be 'yes'/'no')
    for col in ['MTRANS', 'CAEC', 'CALC']:
        train_clean[col] = train_clean[col].astype(str).str.strip().str.capitalize()
        test_clean[col] = test_clean[col].astype(str).str.strip().str.capitalize()
    
    # For binary: ensure lower case and strip
    for col in binary_cols:
        train_clean[col] = train_clean[col].str.strip().str.lower()
        test_clean[col] = test_clean[col].str.strip().str.lower()
    
    
    # Save final cleaned datasets
    train_clean.to_csv(os.path.join(data_dir, 'cleaned_train.csv'), index=False)
    test_clean.to_csv(os.path.join(data_dir, 'cleaned_test.csv'), index=False)
    


    
    import os
    import pandas as pd
    
    # Constants
    data_dir = '/Users/scottiejj/Desktop/AutoKaggle_APAPTED/multi_agents/competition/obesity_risks/'
    train_path = os.path.join(data_dir, 'cleaned_train.csv')
    eda_img_dir = os.path.join(data_dir, 'deep_eda/images')
    os.makedirs(eda_img_dir, exist_ok=True)
    
    # Load cleaned training data (work on a copy)
    train = pd.read_csv(train_path)
    df = train.copy()
    
    # Feature lists (excluding outlier flags and 'id')
    numerical_features = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI'
    ]
    categorical_features = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS'
    ]
    target_col = 'NObeyesdad'
    
    # --- Numerical feature summary by target ---
    print("Numerical feature summary stratified by obesity class:")
    num_summary = target_stratified_numerical_summary(
        df=df,
        numerical_cols=numerical_features,
        target_col=target_col
    )
    print(num_summary)
    
    # --- ANOVA/Kruskal-Wallis significance testing ---
    print("\nANOVA/Kruskal-Wallis test results for numerical features vs. target:")
    anova_results = target_vs_feature_anova(
        df=df,
        numerical_cols=numerical_features,
        target_col=target_col
    )
    print(anova_results)
    
    # --- Categorical feature summary ---
    print("\nCategorical feature summary:")
    cat_summary = summarize_categorical_features(
        df=df,
        categorical_cols=categorical_features
    )
    print(cat_summary)
    
    # --- Chi2 association for categorical features vs. target ---
    print("\nChi2 association between categorical features and target:")
    chi2_results = categorical_vs_target_chi2(
        df=df,
        cat_cols=categorical_features,
        target_col=target_col
    )
    print(chi2_results)
    
    
    # --- BMI segmentation by obesity class (excluding BMI outliers) ---
    print("\nBMI statistics by obesity class (excluding BMI outliers):")
    bmi_segment = segment_bmi_by_obesity_class(
        df=df,
        bmi_col='BMI',
        target_col=target_col,
        outlier_flag_col='BMI_flag'
    )
    print(bmi_segment)
    
    # --- Lifestyle subgroup analysis ---
    lifestyle_cols = ['FAVC', 'SMOKE', 'SCC', 'MTRANS', 'CAEC']
    for col in lifestyle_cols:
        print(f"\nObesity class prevalence by {col}:")
        seg = lifestyle_segment_obesity_rate(
            df=df,
            segment_col=col,
            target_col=target_col,
            min_count=10
        )
        print(seg)
    
    
    import matplotlib.pyplot as plt
    
    # --- Correlation heatmap among numerical features ---
    print("\nGenerating correlation heatmap for numerical features...")
    heatmap_fig = feature_interaction_heatmap(
        df=df,
        numerical_cols=numerical_features,
        method='pearson',
        figsize=(10, 8)
    )
    heatmap_path = os.path.join(eda_img_dir, 'correlation_heatmap_numerical_features.png')
    heatmap_fig.savefig(heatmap_path)
    plt.close(heatmap_fig)
    print(f"Correlation heatmap saved to: {heatmap_path}")
    
    # Optional: Interpretation (text-based)
    corr_matrix = df[numerical_features].corr(method='pearson')
    # Find strongest positive and negative correlations
    corr_pairs = corr_matrix.abs().unstack().sort_values(ascending=False)
    # Exclude self-correlations
    corr_pairs = corr_pairs[corr_pairs < 1]
    # Drop duplicate pairs
    corr_pairs = corr_pairs[~corr_pairs.index.duplicated(keep='first')]
    print("\nStrongest absolute correlations among numerical features:")
    print(corr_pairs.head(5))
    
    
    import matplotlib.pyplot as plt
    
    # --- Boxplots for strongest numerical features (by ANOVA p-value) ---
    signif_num = anova_results.sort_values('p_value').query('significant').head(3)['feature'].tolist()
    for feat in signif_num:
        print(f"\nBoxplot for {feat} stratified by obesity class:")
        fig = feature_vs_target_boxplot(df=df, numerical_col=feat, target_col=target_col, figsize=(8, 5))
        plot_path = os.path.join(eda_img_dir, f'boxplot_{feat}_vs_obesity_class.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Boxplot saved to: {plot_path}")
    
    # --- Optional: Categorical countplot for strongest chi2 feature, if highly significant ---
    # Only do this if fewer than 4 plots have been generated so far
    num_plots = 1 + len(signif_num)  # heatmap + boxplots
    top_cat = chi2_results.sort_values('p_value').query('significant').head(1)
    if not top_cat.empty and num_plots < 4:
        top_cat_feat = top_cat.iloc[0]['feature']
        print(f"\nGenerating countplot for {top_cat_feat} vs. obesity class (top chi2 association)...")
        # We'll use plot_categorical_counts for a single feature, grouped by target
        # But as the tool only plots counts, we use pandas + seaborn here for grouped barplot (allowed for visualization)
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x=top_cat_feat, hue=target_col, ax=ax)
        ax.set_title(f"{top_cat_feat} count by Obesity Class")
        plot_path = os.path.join(eda_img_dir, f'countplot_{top_cat_feat}_by_obesity_class.png')
        fig.savefig(plot_path)
        plt.close(fig)
        print(f"Countplot saved to: {plot_path}")
    


if __name__ == "__main__":
    generated_code_function()