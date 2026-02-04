# 🩺 Breast Cancer Diagnosis Web App

An end-to-end **Machine Learning + Streamlit** web application that predicts whether a breast tumor is **Benign** or **Malignant** based on clinical diagnostic features.  
The app also visualizes feature patterns using a **radar chart** and provides a **confidence score** for each prediction.

---

## 🚀 Project Overview

Early detection of breast cancer is critical in medical diagnosis.  
This project uses a trained **Logistic Regression** model to classify tumors using features derived from cell nuclei measurements.

The system demonstrates the complete ML lifecycle:
- Data preprocessing
- Model training & evaluation
- Model persistence
- Web deployment
- Interactive visualization

---

## 🧠 Machine Learning Details

- **Model**: Logistic Regression 

The machine learning model was evaluated on a held-out test dataset with the following results:
> Accuracy: `98.2%`

> Recall: `97.7%`

- **Preprocessing**: StandardScaler (used during training and inference)
- **Evaluation Focus**: Recall & False Negatives (important for medical diagnosis)
- **Confidence Score**: Probability output from `predict_proba()`

---

## 📊 Features of the Web App

- Sidebar inputs for all diagnostic features  
- Real-time prediction (Benign / Malignant) with confidence score    
- Radar chart visualization (Mean, SE, Worst feature groups)  
- Clean and responsive Streamlit UI  

---

## 🕸️ Radar Chart Visualization

The radar chart helps visualize the **feature profile** of the input sample: Mean, Standard Error(SE), Worst features

Min–Max normalization is applied **only for visualization** to ensure balanced and interpretable radar plots.

---

## 🗂️ Project Structure
```
Breast-Cancer-Diagnosis/
├── app/
│ └── main.py # Streamlit application
│
├── artifacts/
│ └── breast-cancer-model.pkl 
│
├── assets/
│ └── style.css # Custom CSS for UI 
│
├── data/
│ └── breast_cancer.csv 
│
├── notebooks/
│ └── training.ipynb 
│
├── requirements.txt 
└── README.md 
```
