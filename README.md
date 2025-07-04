# Customer Churn Prediction Project
Lloyds Banking Group | Data Science Team

### 📌 Overview
This repository contains a machine learning model to predict customer churn for Lloyds Banking Group. The goal is to identify at-risk customers early so the bank can take proactive retention measures.

### 🔍 Key Features:
✔ Demographic analysis (Age, Income, Marital Status)
✔ Predictive modeling (Random Forest with SMOTE for imbalance handling)
✔ Actionable insights (High-risk customer segments)
✔ Deployment-ready (Saved model + evaluation metrics)

### 🚀 Quick Start
## 1. Install Requirements
bash
pip install -r requirements.txt
(Includes pandas, scikit-learn, imbalanced-learn, matplotlib)

## 2. Run the Analysis
bash
python churn_model.py
Outputs:

model_outputs.json (Performance metrics)

churn_model.pkl (Trained model)

Visualizations (feature_importance.png, confusion_matrix.png)

## 3. Use the Model for Predictions
python
import joblib
model = joblib.load('churn_model.pkl')
predictions = model.predict(new_customer_data)
### 🔍 Key Findings
### 📊 Model Performance
Metric	Score	What It Means
Recall	82%	Correctly flags 82% of churners
Precision	68%	68% of predicted churners are accurate
ROC-AUC	0.87	Strong predictive power
### 🎯 Top Churn Drivers
Young customers (<30)

Low-income individuals

Single marital status

https://feature_importance.png

### 💡 Business Impact
Recommended Actions
Targeted offers for high-risk segments (e.g., financial planning for young, low-income customers).

CRM integration to flag at-risk customers in real time.

Quarterly model updates with new data.

## 📂 Repository Structure
text
├── data/  
│   ├── raw/                  # Original dataset  
│   └── processed/            # Cleaned data  
├── notebooks/  
│   ├── EDA.ipynb             # Exploratory analysis  
│   └── Model_Training.ipynb  # ML pipeline  
├── outputs/  
│   ├── model_outputs.json    # Full metrics  
│   └── visualizations/       # Charts for reports  
├── churn_model.py            # Main training script  
└── requirements.txt          # Python dependencies  
🚧 Limitations & Next Steps
Current limitation: Uses synthetic churn labels (replace with real bank data)

## Future improvements:

Add transaction history/behavioral data

Build an API for real-time predictions

### 🤝 How to Contribute
Fork the repository

Add improvements (e.g., new features, better visualization)

Submit a pull request

# Questions?
Contact: [mohdavaish7860@gmail.com ] | LinkedIn: [https://www.linkedin.com/in/mohd-avaish-706935279/?trk=opento_sprofile_details]
