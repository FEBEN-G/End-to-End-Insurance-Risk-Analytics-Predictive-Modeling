# AlphaCare Insurance Solutions (ACIS) - Final Project Report

## Executive Summary
This project aimed to optimize the marketing strategy for AlphaCare Insurance Solutions by identifying "low-risk" targets and predicting claim severity and premiums in the South African car insurance market. Through Exploratory Data Analysis (EDA), A/B Hypothesis Testing, and Predictive Modeling, we gained significant insights into risk drivers.

## 1. Methodologies

### 1.1 Exploratory Data Analysis (EDA)
- **Data Summarization**: Analyzed ~1 million records from February 2014 to August 2015.
- **Data Quality**: Handled missing values in features like `Bank`, `AccountType`, and `VehicleType`.
- **Visualization**: Used `Seaborn` and `Matplotlib` to visualize distributions of `TotalPremium` and `TotalClaims`.

### 1.2 A/B Hypothesis Testing
- **Metrics**: Claim Frequency (Binary) and Margin (Premium - Claims).
- **Techniques**: Used ANOVA for geographic (Province, ZipCode) analysis and T-Tests for Gender-based comparisons.

### 1.3 Statistical Modeling
- **Preprocessing**: Label encoding for categorical variables and median imputation for numerical data.
- **Algorithms**: Implemented Linear Regression (Baseline) and XGBoost Regressor for Claim Severity and Premium Prediction.
- **Interpretability**: Leveraged SHAP values to explain model predictions.

## 2. Key Findings

### 2.1 Risk Drivers
- **Geographic variation**: We rejected the null hypothesis that risk is uniform across provinces ($p < 0.001$). Provinces like Gauteng show significantly different risk profiles.
- **Gender**: We failed to reject the null hypothesis for gender ($p = 0.84$), suggesting it is not a primary risk differentiator in this data.
- **Vehicle Age**: `RegistrationYear` was identified as a top feature influencing claim amounts.

### 2.2 Model Performance
- **Claim Severity**: The complex nature of claims makes them challenging to predict with simple linear models ($R^2 \approx 0$). XGBoost captured slightly more non-linear patterns.
- **Premium Prediction**: Current premiums are strongly tied to location and vehicle type.

## 3. Recommendations
1.  **Localized Pricing**: Implement regional premium adjustments based on the significant risk differences found across provinces and zip codes.
2.  **Vehicle-Specific Strategies**: Adjust premiums dynamically based on vehicle make, model, and age, as these are strong indicators of claim severity.
3.  **Gender Neutrality**: Maintain gender-neutral pricing for now, as testing showed no significant risk difference.
4.  **Data Enrichment**: Collect more granular data on driver behavior (e.g., telematics) to improve the explanatory power of the predictive models.

## 4. Conclusion
ACIS can significantly enhance its competitive edge by adopting a data-driven, risk-based pricing model. The transition from flat rates to localized, vehicle-specific premiums will attract "low-risk" clients and optimize profitability.
