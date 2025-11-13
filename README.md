# 🩺 Lung Cancer Risk Prediction App

A Streamlit web application for analyzing and predicting lung cancer risk based on patient lifestyle and environmental factors.  
Built with **Python**, **Streamlit**, and **Machine Learning (Scikit-learn & XGBoost)**.

---

## 🚀 Features

### 🧭 1. Exploratory Data Analysis (EDA)
- View dataset structure, summary statistics, and feature correlations  
- Visualize distributions of key factors (e.g., smoking, obesity, air pollution)  
- Automatic **feature importance ranking** using Random Forest

### 🤖 2. Model Training & Evaluation
- Train multiple ML models:  
  - Logistic Regression  
  - Decision Tree  
  - Random Forest  
  - XGBoost  
- Evaluate using **Accuracy**, **Recall**, and **F1-Score**  
- Automatically selects the **best model** based on **medical priority (Recall → F1 → Accuracy)**  
- Feature selection with persistent choices (doesn’t reset between pages)

### 🧪 3. Robustness Test
- Stress-test models with **data noise & outlier injection**  
- Compare model stability under extreme conditions (0.1–1.0 noise level)  
- Identify the **most resilient model** under noisy data scenarios

### 🧮 4. Interactive Prediction
- Input patient conditions manually  
- Predict **lung cancer risk level (Low / Medium / High)** instantly  
- Uses the trained best-performing model  

---

## 🧠 Why Recall & F1 are Prioritized

In medical prediction systems, **accuracy alone can be misleading** because:
- Data is often **imbalanced** (few patients are actually sick)
- A model can achieve 95% accuracy simply by predicting "healthy" for everyone

That’s why this app prioritizes:
- **Recall** → Ensure sick patients are *not missed*  
- **F1-score** → Balance between detecting disease and minimizing false alarms  

> 💡 This approach reduces *false negatives*, which are critical in healthcare prediction.

---

## 🧩 Dataset

**Source:**  
[Kaggle - Cancer Patients and Air Pollution: A New Link](https://www.kaggle.com/datasets/thedevastator/cancer-patients-and-air-pollution-a-new-link)

**Description:**  
The dataset links various factors like smoking, air pollution, and obesity to lung cancer risk levels.  
Each record represents a patient profile with categorized risk levels (`Low`, `Medium`, `High`).

---

## 🧰 Tech Stack

| Component | Technology |
|------------|-------------|
| Frontend | [Streamlit](https://streamlit.io) |
| ML Models | Scikit-learn, XGBoost |
| Visualization | Matplotlib, Seaborn |
| Data Handling | Pandas, NumPy |
| Dataset | Kaggle API |

---
## 🛠️ Streamlit Test
streamlit link : https://cancer-lung-pred.streamlit.app/

---

## ⚙️ Installation & Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/andrianusalvien/lung-cancer-prediction.git
cd lung-cancer-prediction

###

```



