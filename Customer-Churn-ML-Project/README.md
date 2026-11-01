# 📊 Customer Churn Prediction ML System

An end-to-end Machine Learning application designed to analyze customer behavior patterns, predict customer churn, and provide an interactive web interface for real-time predictions.

---

## 📌 Project Overview

Customer churn is a critical metric for subscription-based and service industries. This project focuses on building, evaluating, and deploying a predictive model that identifies customers at high risk of canceling their service, allowing businesses to take proactive retention measures.

---

## 🗂️ Project Structure

```text
Customer-Churn-ML-Project/
│
├── Dataset/                     # Raw and processed dataset files
├── Screenshots/                 # UI and model performance screenshots
├── app.py                       # Streamlit web application for deployment
├── Customer Churn .ipynb        # Jupyter Notebook for EDA, preprocessing & training
├── churn_model.pkl              # Trained classification model
├── model.pkl                    # Core pipeline model artifact
├── model_columns.pkl            # Feature column order for encoding consistency
├── scaler.pkl                   # Feature scaler object (StandardScaler / MinMaxScaler)
└── requirements.txt             # Project dependencies and libraries
