
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure directories exist
os.makedirs('models', exist_ok=True)
os.makedirs('outputs', exist_ok=True)

# 1. Data Exploration
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"Dataset Shape: {df.shape}")
print(f"Class Distribution:\n{df['target'].value_counts()}")

# 2. Data Preprocessing
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 3. Baseline Model (Linear SVM)
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train_scaled, y_train)
y_pred_linear = svm_linear.predict(X_test_scaled)

print("\n--- Linear SVM Results ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred_linear):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_linear):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_linear):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_linear):.4f}")

# 4. Advanced Model (RBF SVM) & 5. Hyperparameter Tuning
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}

grid = GridSearchCV(SVC(probability=True, random_state=42), param_grid, refit=True, verbose=2, cv=5)
grid.fit(X_train_scaled, y_train)

print(f"\nBest Parameters: {grid.best_params_}")
print(f"Best Estimator Score: {grid.best_score_:.4f}")

best_model = grid.best_estimator_
y_pred_best = best_model.predict(X_test_scaled)
y_prob_best = best_model.predict_proba(X_test_scaled)[:, 1]

# 6. Evaluation
print("\n--- Best Model (RBF SVM) Evaluation ---")
print(confusion_matrix(y_test, y_pred_best))
print(classification_report(y_test, y_pred_best))

# 7. ROC Curve & AUC
fpr, tpr, thresholds = roc_curve(y_test, y_prob_best)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Breast Cancer SVM)')
plt.legend(loc="lower right")
plt.savefig('outputs/roc_curve.png')
print(f"ROC Curve saved to outputs/roc_curve.png")
print(f"AUC Score: {roc_auc:.4f}")

# 8. Pipeline & Saving
final_pipeline = Pipeline([
    ('scaler', scaler),
    ('svc', best_model)
])

joblib.dump(final_pipeline, 'models/svm_breast_cancer_model.pkl')
print("Model pipeline saved to models/svm_breast_cancer_model.pkl")

