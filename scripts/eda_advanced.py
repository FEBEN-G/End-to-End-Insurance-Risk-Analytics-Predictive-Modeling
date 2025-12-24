import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
sns.set_theme(style="whitegrid", palette="muted")

def load_and_preprocess(path):
    df = pd.read_csv(path, delimiter='|', low_memory=False)
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    # Handle zero/null premiums for Loss Ratio calculation
    # We'll use a copy to avoid warnings
    df = df.copy()
    df['LossRatio'] = np.where(df['TotalPremium'] > 0, df['TotalClaims'] / df['TotalPremium'], 0)
    return df

def analyze_loss_ratio(df):
    print("Analyzing Loss Ratio variation...")
    
    # 1. By Province
    province_lr = df.groupby('Province')['LossRatio'].mean().sort_values(ascending=False)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=province_lr.index, y=province_lr.values)
    plt.xticks(rotation=45)
    plt.title('Average Loss Ratio by Province')
    plt.ylabel('Average Loss Ratio')
    plt.savefig('reports/loss_ratio_by_province.png')
    plt.close()

    # 2. By Gender
    gender_lr = df.groupby('Gender')['LossRatio'].mean().sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.barplot(x=gender_lr.index, y=gender_lr.values)
    plt.title('Average Loss Ratio by Gender')
    plt.ylabel('Average Loss Ratio')
    plt.savefig('reports/loss_ratio_by_gender.png')
    plt.close()

    # 3. By VehicleType
    vehicle_lr = df.groupby('VehicleType')['LossRatio'].mean().sort_values(ascending=False).head(15)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=vehicle_lr.index, y=vehicle_lr.values)
    plt.xticks(rotation=45)
    plt.title('Top 15 Average Loss Ratio by Vehicle Type')
    plt.ylabel('Average Loss Ratio')
    plt.savefig('reports/loss_ratio_by_vehicle_type.png')
    plt.close()

def analyze_temporal_trends(df):
    print("Analyzing temporal trends...")
    monthly_trends = df.groupby(df['TransactionMonth'].dt.to_period('M')).agg({
        'TotalClaims': 'mean',
        'TotalPremium': 'mean'
    }).reset_index()
    monthly_trends['TransactionMonth'] = monthly_trends['TransactionMonth'].astype(str)
    
    plt.figure(figsize=(14, 7))
    plt.plot(monthly_trends['TransactionMonth'], monthly_trends['TotalClaims'], label='Avg Claims', marker='o')
    plt.plot(monthly_trends['TransactionMonth'], monthly_trends['TotalPremium'], label='Avg Premium', marker='s')
    plt.xticks(rotation=45)
    plt.title('Monthly Average Claims and Premiums')
    plt.legend()
    plt.savefig('reports/temporal_trends.png')
    plt.close()

def analyze_make_model(df):
    print("Analyzing makes and models...")
    make_claims = df.groupby('make')['TotalClaims'].mean().sort_values(ascending=False)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=make_claims.head(10).index, y=make_claims.head(10).values)
    plt.title('Top 10 Vehicle Makes by Average Claim')
    plt.savefig('reports/top_makes_claims.png')
    plt.close()

def creative_plots(df):
    print("Generating creative plots...")
    
    # Plot 1: Correlation Matrix of key financial variables
    plt.figure(figsize=(10, 8))
    cols = ['TotalPremium', 'TotalClaims', 'SumInsured', 'CalculatedPremiumPerTerm', 'LossRatio']
    corr = df[cols].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Correlation Matrix of Financial Variables')
    plt.savefig('reports/creative_correlation_heatmap.png')
    plt.close()
    
    # Plot 2: Boxplot of Claims by Province (Outlier detection)
    plt.figure(figsize=(14, 7))
    # Filter for claims > 0 to see the distribution better
    sns.boxplot(x='Province', y='TotalClaims', data=df[df['TotalClaims'] > 0])
    plt.yscale('log')
    plt.xticks(rotation=45)
    plt.title('Distribution of Positive Claims by Province (Log Scale)')
    plt.savefig('reports/creative_claims_boxplot.png')
    plt.close()
    
    # Plot 3: Scatter plot of Premium vs Claims with Loss Ratio color
    plt.figure(figsize=(12, 8))
    sample = df.sample(min(5000, len(df)))
    scatter = plt.scatter(sample['TotalPremium'], sample['TotalClaims'], 
                        c=sample['LossRatio'], cmap='viridis', alpha=0.5,
                        norm=plt.Normalize(vmin=0, vmax=2))
    plt.colorbar(scatter, label='Loss Ratio')
    plt.xlabel('Total Premium')
    plt.ylabel('Total Claims')
    plt.title('Total Premium vs Total Claims (Colored by Loss Ratio)')
    plt.savefig('reports/creative_premium_claims_scatter.png')
    plt.close()

def main():
    data_path = "/home/feben/Downloads/MachineLearningRating_v3.txt"
    df = load_and_preprocess(data_path)
    
    analyze_loss_ratio(df)
    analyze_temporal_trends(df)
    analyze_make_model(df)
    creative_plots(df)
    
    print("Advanced EDA completed. Outputs saved in 'reports/'.")

if __name__ == "__main__":
    main()
