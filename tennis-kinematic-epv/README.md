# 🎾 Tennis Kinematic EPV Engine & Pursuit Simulator

A hybrid data science and machine learning application that models Expected Point Value (EPV) and defensive interception dynamics in tennis. This project combines differential pursuit equations, dynamic court physics, XGBoost classification, and SHAP (SHapley Additive exPlanations) model interpretability into an interactive Streamlit web dashboard.

---

## 📌 Project Overview

* **Dynamic Interception Physics:** Solves vector pursuit equations to determine defender reachable boundaries, flight times, and spatial deficits based on court friction ($\mu$) and spin angles ($\alpha$).
* **ML Expected Point Value (EPV):** Predicts point-win probabilities using an XGBoost model evaluated against spatial deficit features.
* **Model Interpretability (XAI):** Features real-time SHAP waterfall diagrams and interactive LaTeX log-odds-to-probability transformations for live diagnostic transparency.
* **Surface EPV Heatmap:** Computes dynamic court-wide landing matrices to identify optimal shot placement zones.
* **Extensible CLI Training:** Includes a standalone script (`train_model.py`) to generate a default synthetic model artifact or train on custom player tracking data.

---

## 🛠️ Repository Structure

```text
tennis-kinematic-epv/
│
├── app.py                  # Main Streamlit dashboard code
├── train_model.py          # CLI model training utility
├── sample_data.csv         # Sample CSV layout for custom tracking data
├── epv_xgboost_model.pkl   # Serialized pre-trained XGBoost model artifact
├── requirements.txt        # Python package dependency manifest
├── .gitignore              # Git file ignore list
└── README.md               # Project documentation
