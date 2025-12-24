import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

def load_data(path):
    df = pd.read_csv(path, delimiter='|', low_memory=False)
    return df

def perform_tests(df):
    results = []
    
    # Precompute metrics
    df['Margin'] = df['TotalPremium'] - df['TotalClaims']
    df['HasClaim'] = (df['TotalClaims'] > 0).astype(int)
    
    # 1. H0: There are no risk differences across provinces
    # We'll use ANOVA for Claim Frequency across provinces (as a proxy for risk)
    # Or Chi-squared for HasClaim across Provinces
    print("Testing H0: No risk differences across provinces...")
    provinces = df['Province'].unique()
    province_groups = [df[df['Province'] == p]['HasClaim'] for p in provinces if pd.notnull(p)]
    f_stat, p_val = stats.f_oneway(*province_groups)
    results.append({'Hypothesis': 'No risk differences across provinces', 'Metric': 'Claim Frequency', 'Test': 'ANOVA', 'P-Value': p_val})

    # 2. H0: There are no risk differences between zip codes
    print("Testing H0: No risk differences between zip codes...")
    # Too many zip codes for ANOVA? Let's take Top 10 by volume
    top_zips = df['PostalCode'].value_counts().head(10).index
    zip_groups = [df[df['PostalCode'] == z]['HasClaim'] for z in top_zips]
    f_stat_zip, p_val_zip = stats.f_oneway(*zip_groups)
    results.append({'Hypothesis': 'No risk differences across top zip codes', 'Metric': 'Claim Frequency', 'Test': 'ANOVA', 'P-Value': p_val_zip})

    # 3. H0: There is no significant margin (profit) difference between zip codes
    print("Testing H0: No margin difference between zip codes...")
    zip_margin_groups = [df[df['PostalCode'] == z]['Margin'] for z in top_zips]
    f_stat_margin, p_val_margin = stats.f_oneway(*zip_margin_groups)
    results.append({'Hypothesis': 'No margin difference across top zip codes', 'Metric': 'Margin', 'Test': 'ANOVA', 'P-Value': p_val_margin})

    # 4. H0: There is no significant risk difference between Women and Men
    print("Testing H0: No risk difference between Women and Men...")
    # Gender column cleaning
    gender_df = df[df['Gender'].isin(['Male', 'Female'])]
    male_claims = gender_df[gender_df['Gender'] == 'Male']['HasClaim']
    female_claims = gender_df[gender_df['Gender'] == 'Female']['HasClaim']
    t_stat, p_val_gender = stats.ttest_ind(male_claims, female_claims)
    results.append({'Hypothesis': 'No risk difference between Women and Men', 'Metric': 'Claim Frequency', 'Test': 'T-Test', 'P-Value': p_val_gender})

    return pd.DataFrame(results)

def main():
    data_path = "data/MachineLearningRating_v3.txt"
    df = load_data(data_path)
    
    print("Starting A/B Hypothesis Testing...")
    test_results = perform_tests(df)
    
    print("\nTest Results:")
    print(test_results)
    
    test_results.to_csv('reports/hypothesis_test_results.csv', index=False)
    
    # Interpretation
    print("\n--- Business Recommendations ---")
    for index, row in test_results.iterrows():
        status = "Reject H0" if row['P-Value'] < 0.05 else "Fail to reject H0"
        print(f"{row['Hypothesis']}: {status} (p={row['P-Value']:.4f})")

if __name__ == "__main__":
    main()
