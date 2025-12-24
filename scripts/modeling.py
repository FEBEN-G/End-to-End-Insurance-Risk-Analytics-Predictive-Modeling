import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import shap
import pickle
import os

def load_data(path):
    print(f"Loading data from {path}...")
    df = pd.read_csv(path, delimiter='|', low_memory=False)
    # Select subset of features for modeling
    features = [
        'Province', 'PostalCode', 'Gender', 'MaritalStatus', 
        'make', 'Model', 'VehicleType', 'RegistrationYear', 
        'Cylinders', 'cubiccapacity', 'TotalPremium', 'TotalClaims'
    ]
    df = df[features]
    return df

def preprocess_data(df, target_col):
    df = df.copy()
    
    # Fill missing values
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = df[col].fillna('Unknown')
        else:
            df[col] = df[col].fillna(df[col].median())
            
    # Label Encoding for categorical features
    le = LabelEncoder()
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    X = df.drop(columns=['TotalClaims', 'TotalPremium'])
    y = df[target_col]
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model, X_test, y_test, name):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"\n--- {name} Performance ---")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R2: {r2:.4f}")
    return {'Model': name, 'RMSE': rmse, 'MAE': mae, 'R2': r2}

def main():
    data_path = "data/MachineLearningRating_v3.txt"
    df = load_data(data_path)
    
    # 1. Claim Severity Model (where TotalClaims > 0)
    print("\nTraining Claim Severity Model...")
    df_severe = df[df['TotalClaims'] > 0]
    X_train, X_test, y_train, y_test = preprocess_data(df_severe, 'TotalClaims')
    
    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_res = evaluate_model(lr, X_test, y_test, "Linear Regression (Severity)")
    
    # XGBoost
    xgb_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_res = evaluate_model(xgb_model, X_test, y_test, "XGBoost (Severity)")
    
    # 2. Premium Prediction Model
    print("\nTraining Premium Prediction Model...")
    X_train_p, X_test_p, y_train_p, y_test_p = preprocess_data(df, 'TotalPremium')
    
    xgb_premium = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    xgb_premium.fit(X_train_p, y_train_p)
    prem_res = evaluate_model(xgb_premium, X_test_p, y_test_p, "XGBoost (Premium)")
    
    # Feature Importance with SHAP for Best Model (Premium)
    print("\nPerforming SHAP analysis...")
    explainer = shap.TreeExplainer(xgb_premium)
    shap_values = explainer.shap_values(X_test_p.iloc[:1000])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test_p.iloc[:1000], show=False)
    plt.savefig('reports/shap_summary_premium.png')
    plt.close()
    
    # Save results
    results = pd.DataFrame([lr_res, xgb_res, prem_res])
    results.to_csv('reports/model_evaluation_results.csv', index=False)
    print("\nModeling script completed. Results saved in 'reports/'.")

if __name__ == "__main__":
    main()
