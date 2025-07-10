import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
import os
import plotly.express as px

# Set page config
st.set_page_config(page_title="📦 Return Risk Predictor", layout="centered")

st.title("📦 Return Risk Predictor")
st.write("Predict whether a product will be returned based on review text and metadata.")

# Load models and scaler
try:
    xgb_model = joblib.load("xg_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    lr_model = joblib.load("lr_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# Load feature list for XGBoost
if os.path.exists("features.txt"):
    with open("features.txt", "r") as f:
        xgb_feature_list = [line.strip() for line in f]
else:
    st.error("Missing features.txt file.")
    st.stop()

# Sidebar – Input section
st.sidebar.header("🧾 Input Features")

review_text = st.sidebar.text_area("Review Text", value="The product quality was poor and arrived late.")
delivery_time = st.sidebar.slider("Delivery Time (days)", 1, 15, 5)
score = st.sidebar.slider("Review Score", 1, 5, 3)
category_encoded = st.sidebar.selectbox("Category Code (simulated)", list(range(10)))
helpfulness_ratio = st.sidebar.slider("Helpfulness Ratio", 0.0, 1.0, 0.5)

# Engineered features
review_length = len(review_text)
is_high_rating = 1 if score >= 4 else 0
review_polarity = TextBlob(review_text).sentiment.polarity

# Create input DataFrame
input_raw = pd.DataFrame([{
    'delivery_time': delivery_time,
    'review_polarity': review_polarity,
    'category_encoded': category_encoded,
    'helpfulness_ratio': helpfulness_ratio,
    'review_length': review_length,
    'is_high_rating': is_high_rating,
    'Score': score
}])

# Apply scaling for XGBoost features
input_raw[['delivery_time_scaled', 'review_polarity_scaled']] = scaler.transform(
    input_raw[['delivery_time', 'review_polarity']]
)

# Select model
model_option = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Logistic Regression"])

# Define feature lists
xgb_input = input_raw[xgb_feature_list]

classic_features = ['delivery_time', 'review_polarity', 'category_encoded', 'Score']
classic_input = input_raw[classic_features]

# Select correct model and input
if model_option == "XGBoost":
    model = xgb_model
    X_final = xgb_input
elif model_option == "Random Forest":
    model = rf_model
    X_final = classic_input
elif model_option == "Logistic Regression":
    model = lr_model
    X_final = classic_input
else:
    st.error("Invalid model selection")
    st.stop()

# Predict
try:
    proba = model.predict_proba(X_final)[0][1]
    pred = model.predict(X_final)[0]
except Exception as e:
    st.error(f"Prediction failed: {e}")
    st.stop()

# Output
st.subheader("🔁 Return Risk Prediction")
st.metric("Return Probability", f"{proba:.2f}")
if pred == 1:
    st.error("❌ High Return Risk")
else:
    st.success("✅ Low Return Risk")

# Show Feature Importance for XGBoost
if model_option == "XGBoost":
    st.subheader("📊 Feature Importance (XGBoost)")
    importance = model.feature_importances_
    fi_df = pd.DataFrame({'Feature': xgb_feature_list, 'Importance': importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=True)
    fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig)
