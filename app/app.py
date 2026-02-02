import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load('models/svm_breast_cancer_model.pkl')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}. Please ensure 'models/svm_breast_cancer_model.pkl' exists.")
    st.stop()

st.title("Breast Cancer Classification System")
st.markdown("Classify tumors as Malignant or Benign using SVM.")

st.sidebar.header("Input Features (Mean)")
radius_mean = st.sidebar.slider('Radius Mean', 5.0, 30.0, 14.0)
texture_mean = st.sidebar.slider('Texture Mean', 9.0, 40.0, 19.0)
perimeter_mean = st.sidebar.slider('Perimeter Mean', 40.0, 190.0, 90.0)
area_mean = st.sidebar.slider('Area Mean', 140.0, 2500.0, 650.0)
smoothness_mean = st.sidebar.slider('Smoothness Mean', 0.05, 0.20, 0.1)
compactness_mean = st.sidebar.slider('Compactness Mean', 0.02, 0.40, 0.1)
concavity_mean = st.sidebar.slider('Concavity Mean', 0.0, 0.50, 0.1)
concave_points_mean = st.sidebar.slider('Concave Points Mean', 0.0, 0.20, 0.05)
symmetry_mean = st.sidebar.slider('Symmetry Mean', 0.10, 0.35, 0.2)
fractal_dimension_mean = st.sidebar.slider('Fractal Dimension Mean', 0.05, 0.10, 0.06)

# Construct feature array with defaults for others
# Using mean values from dataset for SE and Worst features to simplify demo
# (In a real app, you'd want inputs for all or a more sophisticated imputation)
defaults = [0.4, 1.2, 2.8, 40.0, 0.007, 0.025, 0.03, 0.01, 0.02, 0.003, # SE
            16.2, 25.6, 107.0, 880.0, 0.13, 0.25, 0.27, 0.11, 0.29, 0.08] # Worst

features_list = [radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, 
                 compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean]
features_list.extend(defaults)

input_features = np.array(features_list).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_features)
    proba = model.predict_proba(input_features)
    
    label = "Benign" if prediction[0] == 1 else "Malignant" # 0=Malignant, 1=Benign in sklearn dataset usually? Wait, let's verify.
    # In sklearn breast_cancer:
    # 0 = Malignant
    # 1 = Benign
    # The prompt asks to classify.
    
    st.subheader(f"Result: {label}")
    st.write(f"Confidence: {np.max(proba):.2f}")
