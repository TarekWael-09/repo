
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer

# Load the model
try:
    model = joblib.load('breast_cancer_model.pkl')
except:
    st.error("Model file not found. Please run the training script first.")

data = load_breast_cancer()

st.set_page_config(page_title="Breast Cancer Diagnostic Tool", layout="wide")

st.title("🎗️ Breast Cancer Prediction Dashboard")
st.markdown("""
This application predicts whether a tumor is **Malignant** or **Benign** using a Random Forest Classifier.
""")

st.sidebar.header("Input Tumor Features")

def user_input_features():
    inputs = {}
    # Key features for UI
    for name in data.feature_names[:10]:
        inputs[name] = st.sidebar.number_input(f"Enter {name}", value=float(data.data[:, data.feature_names.tolist().index(name)].mean()))
    
    # Padding remaining features with mean values to match model input (30 features)
    for name in data.feature_names[10:]:
        inputs[name] = data.data[:, data.feature_names.tolist().index(name)].mean()
        
    return pd.DataFrame(inputs, index=[0])

df_input = user_input_features()

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Selected Parameters (Top 10)")
    st.write(df_input.iloc[:, :10])

with col2:
    st.subheader("Prediction Result")
    if st.button("Predict"):
        prediction = model.predict(df_input)
        prediction_proba = model.predict_proba(df_input)
        
        if prediction[0] == 0:
            st.error("Result: Malignant (Cancerous)")
        else:
            st.success("Result: Benign (Non-Cancerous)")
            
        st.write(f"**Confidence:** {np.max(prediction_proba)*100:.2f}%")

st.info("Note: This tool is for educational purposes.")
