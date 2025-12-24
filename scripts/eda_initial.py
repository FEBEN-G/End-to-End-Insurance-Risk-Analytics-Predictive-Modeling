import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style for plots
sns.set_theme(style="whitegrid")

def load_data(path):
    print(f"Loading data from {path}...")
    # Delimiter is '|' based on peek
    df = pd.read_csv(path, delimiter='|', low_memory=False)
    print(f"Data loaded. Shape: {df.shape}")
    return df

def basic_eda(df):
    print("\n--- Basic Information ---")
    print(df.info())
    
    print("\n--- Descriptive Statistics ---")
    print(df.describe())
    
    print("\n--- Missing Values ---")
    print(df.isnull().sum()[df.isnull().sum() > 0])

def preprocess_data(df):
    print("\nPreprocessing data...")
    # Convert dates
    df['TransactionMonth'] = pd.to_datetime(df['TransactionMonth'])
    
    # Calculate Loss Ratio
    # Replace zeros in TotalPremium with small value to avoid division by zero or handle separately
    df['LossRatio'] = df['TotalClaims'] / df['TotalPremium']
    
    return df

def plot_distributions(df):
    print("Plotting distributions...")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    sns.histplot(df['TotalPremium'], bins=50, ax=axes[0], kde=True)
    axes[0].set_title('Distribution of Total Premium')
    
    sns.histplot(df['TotalClaims'], bins=50, ax=axes[1], kde=True)
    axes[1].set_title('Distribution of Total Claims')
    
    plt.tight_layout()
    plt.savefig('reports/distributions.png')
    plt.close()

def main():
    data_path = "/home/feben/Downloads/MachineLearningRating_v3.txt"
    df = load_data(data_path)
    
    basic_eda(df)
    df = preprocess_data(df)
    
    # Save a small sample for faster local analysis if needed
    # df.sample(10000).to_csv('data/sample_data.csv', index=False)
    
    plot_distributions(df)
    
    print("\nEDA script completed. Check 'reports/' for outputs.")

if __name__ == "__main__":
    main()
