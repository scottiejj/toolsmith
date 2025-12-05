import sys
import os
import importlib.util
import sys as _sys

# Basic paths
sys.path.extend(['.', '..', '../..', '../../..', '../../../..', 'multi_agents','multi_agents/tools', 'multi_agents/prompts'])
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

phase_module_path = os.path.join('multi_agents', 'competition', "obesity_risks", "model_build_predict", "model_build_predict_generated_tools", "model_build_predict.py")

spec = importlib.util.spec_from_file_location('phase_tools', phase_module_path)
phase_tools = importlib.util.module_from_spec(spec)
_sys.modules['phase_tools'] = phase_tools
spec.loader.exec_module(phase_tools)
from phase_tools import *  

def generated_code_function():
    import numpy as np
    import pandas as pd
phase_module_path = os.path.join('multi_agents', 'competition', "obesity_risks", "feature_engineering", "feature_engineering_generated_tools", "feature_engineering.py")

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
    
    # File paths
    data_dir = '/Users/scottiejj/Desktop/AutoKaggle_APAPTED/multi_agents/competition/obesity_risks/'
    train_path = os.path.join(data_dir, 'cleaned_train.csv')
    test_path = os.path.join(data_dir, 'cleaned_test.csv')
    
    # Load data (work on copies)
    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    train_fe = train.copy()
    test_fe = test.copy()
    
    # 1. BMI Category
    train_fe = create_bmi_category_feature(train_fe, bmi_col='BMI', new_col='BMI_category')
    test_fe = create_bmi_category_feature(test_fe, bmi_col='BMI', new_col='BMI_category')
    
    # 2. Lifestyle Risk Score
    train_fe = create_lifestyle_risk_score(
        train_fe,
        favc_col='FAVC', caec_col='CAEC', calc_col='CALC', scc_col='SCC', smoke_col='SMOKE',
        fcvc_col='FCVC', ch2o_col='CH2O', faf_col='FAF', tue_col='TUE', new_col='lifestyle_risk_score'
    )
    test_fe = create_lifestyle_risk_score(
        test_fe,
        favc_col='FAVC', caec_col='CAEC', calc_col='CALC', scc_col='SCC', smoke_col='SMOKE',
        fcvc_col='FCVC', ch2o_col='CH2O', faf_col='FAF', tue_col='TUE', new_col='lifestyle_risk_score'
    )
    
    # 3. Age-Lifestyle Interaction
    train_fe = create_age_lifestyle_interaction(
        train_fe,
        age_col='Age', favc_col='FAVC', fcvc_col='FCVC', faf_col='FAF', new_col='age_lifestyle_interaction'
    )
    test_fe = create_age_lifestyle_interaction(
        test_fe,
        age_col='Age', favc_col='FAVC', fcvc_col='FCVC', faf_col='FAF', new_col='age_lifestyle_interaction'
    )
    
    # 4. BMI-Weight Interaction
    train_fe = create_bmi_weight_interaction(
        train_fe, bmi_col='BMI', weight_col='Weight', new_col='bmi_weight_interaction'
    )
    test_fe = create_bmi_weight_interaction(
        test_fe, bmi_col='BMI', weight_col='Weight', new_col='bmi_weight_interaction'
    )
    
    # 5. Lifestyle Balance
    train_fe = create_lifestyle_balance_feature(
        train_fe, faf_col='FAF', tue_col='TUE', new_col='lifestyle_balance'
    )
    test_fe = create_lifestyle_balance_feature(
        test_fe, faf_col='FAF', tue_col='TUE', new_col='lifestyle_balance'
    )
    
    
    # Define categorical columns to encode (exclude 'id' and target)
    categorical_cols = [
        'Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE',
        'SCC', 'CALC', 'MTRANS', 'BMI_category'
    ]
    
    train_fe = one_hot_encode_columns(train_fe, categorical_cols=categorical_cols, drop_original=False)
    test_fe = one_hot_encode_columns(test_fe, categorical_cols=categorical_cols, drop_original=False)
    
    
    # Columns to standardize
    numerical_cols = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI',
        'lifestyle_risk_score', 'age_lifestyle_interaction', 'bmi_weight_interaction', 'lifestyle_balance'
    ]
    
    train_fe = standardize_numerical_features(train_fe, numerical_cols=numerical_cols, suffix="_std")
    test_fe = standardize_numerical_features(test_fe, numerical_cols=numerical_cols, suffix="_std")
    
    # Aggregate flag columns (all ending with '_flag')
    flag_cols = [col for col in train_fe.columns if col.endswith('_flag')]
    train_fe = aggregate_flag_features(train_fe, flag_cols=flag_cols, new_col="n_flags")
    test_fe = aggregate_flag_features(test_fe, flag_cols=flag_cols, new_col="n_flags")
    
    
    # Build output feature list
    
    # All columns to exclude
    flag_cols = [col for col in train_fe.columns if col.endswith('_flag')]
    exclude_cols = set(flag_cols)
    
    # These always to be retained
    always_keep = ['id', 'n_flags']
    if 'NObeyesdad' in train_fe.columns:
        always_keep.append('NObeyesdad')
    
    # Identify all one-hot columns (start with any of the categorical col names plus '_')
    one_hot_prefixes = [c + '_' for c in categorical_cols]
    one_hot_cols = [col for col in train_fe.columns for prefix in one_hot_prefixes if col.startswith(prefix)]
    
    # Collect all standardized columns
    std_cols = [col for col in train_fe.columns if col.endswith('_std')]
    
    # Engineered columns
    engineered_cols = [
        'BMI_category', 'lifestyle_risk_score', 'age_lifestyle_interaction',
        'bmi_weight_interaction', 'lifestyle_balance'
    ]
    
    # Original numerical columns (excluding any dropped)
    original_num_cols = [
        'Age', 'Height', 'Weight', 'FCVC', 'NCP', 'CH2O', 'FAF', 'TUE', 'BMI'
    ]
    
    # Categorical columns (for possible tree-based models, but also one-hot exists)
    categorical_keep = categorical_cols
    
    # Final list (order: id, raw num, engineered, std, one-hot, n_flags, target)
    feature_list = (
        ['id']
        + original_num_cols
        + engineered_cols
        + std_cols
        + one_hot_cols
        + ['n_flags']
    )
    if 'NObeyesdad' in train_fe.columns:
        feature_list += ['NObeyesdad']
    
    # Remove any duplicates
    feature_list = list(dict.fromkeys(feature_list))
    
    # Remove any columns in exclude_cols
    feature_list = [col for col in feature_list if col not in exclude_cols]
    
    # Filter to only columns present in both train and test (except target)
    train_keep = [col for col in feature_list if col in train_fe.columns]
    test_keep = [col for col in feature_list if col in test_fe.columns and col != 'NObeyesdad']
    
    # Align columns
    train_out = train_fe[train_keep].copy()
    test_out = test_fe[test_keep].copy()
    
    # Save outputs
    processed_train_path = os.path.join(data_dir, 'processed_train.csv')
    processed_test_path = os.path.join(data_dir, 'processed_test.csv')
    train_out.to_csv(processed_train_path, index=False)
    test_out.to_csv(processed_test_path, index=False)
    
    print("Feature engineering complete.")
    print(f"Processed train shape: {train_out.shape}")
    print(f"Processed test shape: {test_out.shape}")
    print("Included features:", train_keep)
    


    
    import os
    import pandas as pd
    
    # File paths
    data_dir = '/Users/scottiejj/Desktop/AutoKaggle_APAPTED/multi_agents/competition/obesity_risks/'
    train_file = os.path.join(data_dir, 'processed_train.csv')
    test_file = os.path.join(data_dir, 'processed_test.csv')
    
    # Load data (always work on copies)
    train_df = pd.read_csv(train_file)
    test_df = pd.read_csv(test_file)
    train = train_df.copy()
    test = test_df.copy()
    
    # Remove non-feature columns
    drop_cols = ['id', 'NObeyesdad', 'BMI_category']  # 'BMI_category' is string, don't use; keep one-hot BMI_category_* columns
    all_cols = train.columns.tolist()
    # Identify one-hot columns (all starting with categorical col names plus '_')
    one_hot_bmi = [col for col in all_cols if col.startswith('BMI_category_')]
    # All one-hots are already in, so dropping 'BMI_category' is safe
    
    # Build feature list
    X_cols = [c for c in all_cols if c not in drop_cols]
    # Remove any columns that have object or string type (other than one-hot columns)
    X_cols = [c for c in X_cols if (train[c].dtype != 'object' or c in one_hot_bmi)]
    
    # Ensure test has the same columns
    X_test_cols = [c for c in X_cols if c in test.columns]
    
    # Final modeling matrices
    X_train = train[X_test_cols].copy()
    y_train = train['NObeyesdad'].copy()
    X_test = test[X_test_cols].copy()
    
    # Print final features and types
    print("Final modeling feature columns:")
    for col in X_train.columns:
        print(f"{col}: {X_train[col].dtype}")
    print(f"\nX_train shape: {X_train.shape}, X_test shape: {X_test.shape}, y_train shape: {y_train.shape}")
    print("Unique target classes:", sorted(y_train.unique()))
    
    # Define class labels (in order)
    target_labels = sorted(y_train.unique())
    
    model_results = []
    
    # Logistic Regression
    lr_result = fit_and_evaluate_logistic_regression(
        X=X_train,
        y=y_train,
        feature_cols=X_train.columns.tolist(),
        target_labels=target_labels,
        cv=5,
        random_state=42,
        solver='liblinear',
        max_iter=500,
        return_estimator=True
    )
    model_results.append(lr_result)
    
    # Random Forest
    rf_result = fit_and_evaluate_random_forest(
        X=X_train,
        y=y_train,
        feature_cols=X_train.columns.tolist(),
        target_labels=target_labels,
        cv=5,
        n_estimators=100,
        random_state=42,
        return_estimator=True
    )
    model_results.append(rf_result)
    
    # SVC
    svc_result = fit_and_evaluate_svc(
        X=X_train,
        y=y_train,
        feature_cols=X_train.columns.tolist(),
        target_labels=target_labels,
        cv=5,
        kernel='rbf',
        C=1.0,
        gamma='scale',
        random_state=42,
        return_estimator=True
    )
    model_results.append(svc_result)
    
    # Summarize results
    import pandas as pd
    
    summary_data = []
    for res in model_results:
        summary_data.append({
            "Model": res['model_name'],
            "CV Accuracy (mean)": f"{res['mean_scores']['accuracy']:.4f} ± {res['std_scores']['accuracy']:.4f}",
            "CV Macro-F1 (mean)": f"{res['mean_scores'].get('macro_f1', 0.0):.4f} ± {res['std_scores'].get('macro_f1', 0.0):.4f}",
            "# Features": res['n_features'],
            "# Samples": res['n_samples']
        })
    summary_df = pd.DataFrame(summary_data)
    print("\nCross-Validation Results Summary:")
    print(summary_df)
    
    import numpy as np
    
    # Find best model(s)
    accs = [res['mean_scores']['accuracy'] for res in model_results]
    best_idx = np.argmax(accs)
    best_model = model_results[best_idx]
    print(f"\nSelected model for final prediction: {best_model['model_name']} (Accuracy: {best_model['mean_scores']['accuracy']:.4f})")
    
    # Check for close second (within 0.5% absolute accuracy)
    close_idxs = [i for i, acc in enumerate(accs) if abs(acc - accs[best_idx]) < 0.005 and i != best_idx]
    ensemble_used = False
    
    if close_idxs:
        print("Two models are very close in performance; using soft-voting ensemble.")
        estimators = [model_results[best_idx]['estimator'], model_results[close_idxs[0]]['estimator']]
        preds = soft_voting_ensemble_predict(
            estimators=estimators,
            X=X_test,
            class_labels=target_labels
        )
        ensemble_used = True
    else:
        # Use the best model
        estimator = best_model['estimator']
        # If estimator supports predict_proba, use calibrated prediction
        if hasattr(estimator, 'predict_proba'):
            y_probs = estimator.predict_proba(X_test)
            preds = obesity_prediction_calibrator(
                y_probs=y_probs,
                class_labels=target_labels,
                method='argmax'
            )
        else:
            preds = estimator.predict(X_test)
    
    # Print class distribution
    preds_series = pd.Series(preds)
    print("\nTest set prediction class distribution:")
    print(preds_series.value_counts().sort_index())
    
    # Prepare submission
    ids = test_df['id']
    submission = obesity_submission_formatter(
        ids=ids,
        preds=preds,
        id_col='id',
        target_col='NObeyesdad'
    )
    
    # Sanity checks
    assert submission.shape[0] == X_test.shape[0], "Submission row count does not match test set."
    assert set(submission.columns) == {'id', 'NObeyesdad'}, "Submission columns incorrect."
    print("\nSubmission file preview:")
    print(submission.head())
    print("\nSubmission class value counts:")
    print(submission['NObeyesdad'].value_counts())
    
    # Save to file
    submission_path = os.path.join(data_dir, 'submission.csv')
    submission.to_csv(submission_path, index=False)
    print(f"\nSubmission file saved to: {submission_path}")
    


if __name__ == "__main__":
    generated_code_function()