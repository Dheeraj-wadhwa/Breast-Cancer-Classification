# 🩺 Breast Cancer Classification using SVM

An end-to-end Machine Learning project for classifying breast tumors as **Malignant** or **Benign** using Support Vector Machines (SVM). This project demonstrates industry-standard ML practices including data preprocessing, hyperparameter tuning, model evaluation, and deployment.

## 📊 Project Overview

This project uses the **sklearn Breast Cancer dataset** (569 samples, 30 features) to build a high-performance classification system achieving **~99% AUC score**. All analysis and model generation are contained within a comprehensive Jupyter Notebook.

### Key Features
- ✅ Complete data exploration and preprocessing pipeline
- ✅ Baseline Linear SVM and advanced RBF kernel implementation
- ✅ Hyperparameter tuning using GridSearchCV
- ✅ Comprehensive model evaluation (Confusion Matrix, ROC-AUC)
- ✅ Production-ready model pipeline with joblib serialization
- ✅ Interactive Streamlit web application for predictions

## 🗂️ Project Structure

```
SVM/
├── notebooks/
│   ├── breast_cancer_svm.ipynb    # Main analysis and training notebook
│   ├── svm_breast_cancer_model.pkl # Saved model pipeline
│   └── roc_curve.png              # ROC curve visualization
├── app/
│   └── app.py                     # Streamlit deployment app
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### Installation

1. Clone or navigate to the project directory:
```bash
cd c:\Users\lenovo\SVM
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 🔬 Usage

### 1. Run the Analysis & Training
All training and evaluation logic is contained in the Jupyter notebook. Running the cells will generate the model and visualizations.

```bash
jupyter notebook notebooks/breast_cancer_svm.ipynb
```

**Outputs generated in `notebooks/`:**
- `svm_breast_cancer_model.pkl`: The serialized model pipeline.
- `roc_curve.png`: The ROC-AUC curve plot.

### 2. Run the Streamlit App
Once the model file is generated, you can launch the interactive web application:

```bash
streamlit run app/app.py
```

The app allows you to:
- Input tumor feature values via sliders
- Get real-time predictions (Malignant/Benign)
- View confidence scores

## 📈 Model Performance

| Metric | Score |
|--------|-------|
| **AUC** | ~0.99 |
| **Accuracy** | ~0.98 |
| **Precision** | ~0.98 |
| **Recall** | ~0.98 |
| **F1-Score** | ~0.98 |

## 👤 Author
**Dheeraj Wadhwa**

---

**⭐ If this project helped you, please star it!**
