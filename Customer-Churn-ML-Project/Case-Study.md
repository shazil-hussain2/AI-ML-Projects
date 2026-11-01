# 📈 Case Study: Reducing Customer Churn via Predictive Machine Learning & Real-Time Inference

## 1. Executive Summary
Customer retention is a top strategic priority in subscription-based and service industries, as acquiring a new customer costs significantly more than retaining an existing one. This case study evaluates an end-to-end Machine Learning pipeline developed to identify high-risk churn customers. By combining exploratory data analysis, predictive modeling, model artifact serialization, and an interactive Streamlit UI, this solution enables customer success teams to take proactive retention measures before account cancellation occurs.

---

## 2. Business Problem Statement
* **Challenge:** High churn rates lead to recurring revenue leakage and increased acquisition overheads.
* **Goal:** Build an automated binary classification model to predict customer churn probability and deploy a real-time web interface for inference.
* **Impact:** Provide actionable customer-level insights to trigger targeted retention offers, reducing overall churn and improving Customer Lifetime Value (CLV).

---

## 3. Data Architecture & Pipeline
The technical workflow is divided into four main stages:

```text
+-------------------+      +-------------------------+      +-----------------------+      +-------------------+
|  Raw Data Layer   | ---> | Notebook Exploration    | ---> | Model Serialization   | ---> |   Deployment UI   |
| (Dataset/ folder) |      | (Customer Churn .ipynb) |      | (.pkl Artifacts)      |      |     (app.py)      |
+-------------------+      +-------------------------+      +-----------------------+      +-------------------+
