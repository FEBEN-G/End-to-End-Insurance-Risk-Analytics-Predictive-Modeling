# End-to-End Insurance Risk Analytics & Predictive Modeling

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![DVC](https://img.shields.io/badge/DVC-Data_Versioning-orange.svg)](https://dvc.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Machine_Learning-green.svg)](https://xgboost.readthedocs.io/)

## 📌 Project Overview
As part of AlphaCare Insurance Solutions (ACIS), this project focuses on developing cutting-edge risk and predictive analytics for car insurance in South Africa. The primary objective is to optimize marketing strategies and discover "low-risk" targets for premium reduction, thereby attracting new clients.

## 🚀 Key Features
- **Exploratory Data Analysis (EDA)**: Comprehensive analysis of historical claim data (Feb 2014 - Aug 2015).
- **A/B Hypothesis Testing**: Statistical validation of risk drivers across provinces, zip codes, and demographics.
- **Predictive Modeling**: Development of machine learning models to predict claim severity and optimize premiums.
- **Explainable AI (XAI)**: SHAP analysis to provide transparency and business insights into model predictions.
- **Reproducibility**: Data versioning using DVC.

## 📁 Project Structure
```bash
├── data/               # Tracked datasets (DVC)
├── scripts/            # Modular Python scripts for EDA, Testing, and Modeling
│   ├── eda_initial.py
│   ├── eda_advanced.py
│   ├── hypothesis_testing.py
│   └── modeling.py
├── reports/            # Visualizations and evaluation results
├── notebook/           # Jupyter notebooks for interactive analysis
├── venv/               # Virtual environment
├── FINAL_REPORT.md     # Comprehensive findings and recommendations
└── README.md           # Project documentation
```

## 🛠️ Tech Stack
- **Languages**: Python
- **Libraries**: Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Scipy, XGBoost, SHAP
- **Tools**: Git, GitHub, DVC (Data Version Control)

## 📊 Key Findings
- **Regional Risk**: Significant risk differences were identified across South African provinces ($p < 0.05$).
- **Gender Insight**: Statistical testing failed to reject the null hypothesis for gender risk differences ($p = 0.84$).
- **Primary Drivers**: Location (ZipCode) and Vehicle Age are the most influential factors in determining claim severity.

## ⚙️ Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone https://github.com/FEBEN-G/End-to-End-Insurance-Risk-Analytics-Predictive-Modeling.git
   cd End-to-End-Insurance-Risk-Analytics-Predictive-Modeling
   ```

2. **Setup Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt  # Or manually install pandas, numpy, seaborn, xgboost, shap, scipy
   ```

3. **Run Analysis**:
   ```bash
   python scripts/eda_advanced.py
   python scripts/hypothesis_testing.py
   python scripts/modeling.py
   ```

## 📝 Recommendations
- Implement **region-specific pricing** adjustments.
- Optimize premiums based on **Vehicle Age** and **Make/Model**.
- Maintain **gender-neutral** pricing strategies based on data evidence.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is licensed under the MIT License.
