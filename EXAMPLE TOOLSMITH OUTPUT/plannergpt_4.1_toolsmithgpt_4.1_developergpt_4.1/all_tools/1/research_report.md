# COMPETITION RESEARCH REPORT

## 1. PRELIMINARY EDA
During the preliminary exploratory data analysis (EDA), we examined the dataset's structure, focusing on both numerical and categorical features. Key findings included:
- The target variable `NObeyesdad` showed distinct distributions across obesity classes, with correlations identified between various lifestyle factors and obesity levels.
- No missing values were present in the dataset, indicating good data integrity.
- Outliers were detected in numerical features, particularly `Age`, `Height`, and `Weight`, which could skew analyses and model performance.

Based on these findings, no features were created or deleted, but the presence of outliers indicated a need for cleaning in the next phase.

## 2. DATA CLEANING
The data cleaning process involved several critical actions:
- All numerical features were converted to the appropriate type and filled with median values where invalid entries were detected, ensuring consistency.
- Categorical features had invalid values replaced with the mode, and string values were normalized to ensure consistency across the dataset.
- Outliers were flagged and managed through capping or dropping extreme values based on the interquartile range (IQR) method.

These steps were crucial to prepare the dataset for further analysis, addressing issues that could hinder model performance.

## 3. DEEP EDA
In-depth exploratory data analysis revealed important relationships between features:
- Numerical features were analyzed for their statistical significance in relation to the obesity classes, revealing clear trends in `Age`, `Weight`, and `BMI`.
- Correlation analyses highlighted strong relationships between `Weight` and `BMI`, indicating potential redundancy.
- Insights into lifestyle factors such as physical activity (`FAF`) showed weak negative correlations with BMI, suggesting areas for intervention.

These findings informed subsequent data cleaning and feature engineering decisions, emphasizing the importance of specific lifestyle factors in predicting obesity.

## 4. FEATURE ENGINEERING
Feature engineering involved creating and modifying variables to enhance model performance:
- New features such as `BMI_category`, `lifestyle_risk_score`, and interaction terms (e.g., `age_lifestyle_interaction`) were introduced to capture non-linear relationships and demographic interactions.
- Standardization of numerical features was performed to ensure they were on a similar scale, improving model convergence.
- Categorical features underwent one-hot encoding to facilitate inclusion in predictive models.

The rationale for these actions was to enrich the model's input space, allowing it to capture complex relationships more effectively.

## 5. MODEL BUILDING, VALIDATION, AND PREDICTION
Three models were trained: Logistic Regression, Random Forest, and Support Vector Classifier (SVC). Their performance metrics were evaluated using cross-validation:
- Each model's mean accuracy and macro-F1 score were computed, with the best-performing model selected based on these criteria.
- Care was taken to ensure consistency in feature selection and encoding between the training and test datasets.

The predicted class distribution of the `NObeyesdad` variable was analyzed, comparing it against training data distributions to assess model alignment with expected outcomes.

## 6. CONCLUSION
The research process highlighted the importance of thorough data exploration, cleaning, and feature engineering in building robust predictive models. Key insights included the impact of lifestyle factors on obesity and the utility of interaction terms in capturing complex relationships. Future modeling efforts should continue to emphasize consistency and feature relevance while considering ensemble methods to enhance performance.