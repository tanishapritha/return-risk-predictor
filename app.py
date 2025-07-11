import streamlit as st
import pandas as pd
import numpy as np
from textblob import TextBlob
import joblib
import os
import plotly.express as px

st.set_page_config(page_title="📦 Return Risk Predictor", layout="centered")

st.title("📦 Return Risk Predictor")

# Load models and scaler
try:
    xgb_model = joblib.load("xg_model.pkl")
    rf_model = joblib.load("rf_model.pkl")
    lr_model = joblib.load("lr_model.pkl")
    scaler = joblib.load("scaler.pkl")
except Exception as e:
    st.error(f"❌ Error loading model files: {e}")
    st.stop()

# Load XGBoost feature list
if os.path.exists("features.txt"):
    with open("features.txt", "r") as f:
        xgb_feature_list = [line.strip() for line in f]
else:
    st.error("Missing features.txt file.")
    st.stop()

# Moved model selection before tabs
model_option = st.selectbox("Select Model", ["XGBoost", "Random Forest", "Logistic Regression"])

# Tabs for Prediction & Insights
tabs = st.tabs(["🧪 Make a Prediction", "🧠 Model Insights"])

# ========== 🧪 TAB 1: PREDICTION ==========
with tabs[0]:
    st.header("🧪 Make a Prediction")

    # Sidebar input
    st.sidebar.header("🧾 Input Features")

    review_text = st.sidebar.text_area("Review Text", value="The product quality was poor and arrived late.")
    delivery_time = st.sidebar.slider("Delivery Time (days)", 1, 15, 5)
    score = st.sidebar.slider("Review Score", 1, 5, 3)
    category_encoded = st.sidebar.selectbox("Category Code (simulated)", list(range(10)))
    helpfulness_ratio = st.sidebar.slider("Helpfulness Ratio", 0.0, 1.0, 0.5)

    # Feature engineering
    review_length = len(review_text)
    is_high_rating = 1 if score >= 4 else 0
    review_polarity = TextBlob(review_text).sentiment.polarity

    input_raw = pd.DataFrame([{
        'delivery_time': delivery_time,
        'review_polarity': review_polarity,
        'category_encoded': category_encoded,
        'helpfulness_ratio': helpfulness_ratio,
        'review_length': review_length,
        'is_high_rating': is_high_rating,
        'Score': score
    }])

    # Apply scaling
    input_raw[['delivery_time_scaled', 'review_polarity_scaled']] = scaler.transform(
        input_raw[['delivery_time', 'review_polarity']]
    )

    # Model description
    st.info({
        "XGBoost": """
### 🔍 XGBoost (Extreme Gradient Boosting)
A powerful tree-based model trained in steps. Each small tree corrects mistakes from the previous ones. Ideal for structured data like ours.
""",
        "Random Forest": """
### 🌲 Random Forest
Builds many decision trees on different data slices and combines them. Reduces overfitting and improves stability.
""",
        "Logistic Regression": """
### 📈 Logistic Regression
A linear model that predicts probabilities based on weighted features. Simple and interpretable.
"""
    }[model_option])

    # Prepare input
    xgb_input = input_raw[xgb_feature_list]
    classic_features = ['delivery_time', 'review_polarity', 'category_encoded', 'Score']
    classic_input = input_raw[classic_features]

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

    st.subheader("🔁 Return Risk Prediction")
    st.metric("Return Probability", f"{proba:.2f}")
    if pred == 1:
        st.error("❌ High Return Risk")
    else:
        st.success("✅ Low Return Risk")

    # Feature Importance for XGBoost
    if model_option == "XGBoost":
        st.subheader("📊 Feature Importance (XGBoost)")
        importance = model.feature_importances_
        fi_df = pd.DataFrame({'Feature': xgb_feature_list, 'Importance': importance})
        fi_df = fi_df.sort_values(by="Importance", ascending=True)
        fig = px.bar(fi_df, x='Importance', y='Feature', orientation='h')
        st.plotly_chart(fig)

# ========== 🧠 TAB 2: INSIGHTS ==========
with tabs[1]:
    st.header(f"🧠 How {model_option} Works")

    helpfulness_note = """
**What is `helpfulness_ratio`?**

This feature shows how useful other users found a review.

🧮 **Formula:**  
$$
\\text{helpfulness\\_ratio} = \\frac{\\text{HelpfulnessNumerator}}{\\text{HelpfulnessDenominator}}
$$

- If 3 out of 4 people found the review helpful → 3/4 = 0.75  
- If no one voted (0/0), we set the ratio to 0

Reviews marked helpful by users tend to be more reliable and often indicate stronger signals for return risk.
"""

    if model_option == "XGBoost":
        st.markdown("""
### 🌳 XGBoost (Extreme Gradient Boosting)
XGBoost builds a series of small decision trees, each one trying to fix the mistakes made by the last. It’s like an “error-correcting team” of trees.

**Why it works well:**
- Fast and efficient
- Handles both numeric and categorical data
- Highlights which features were most influential

**Features used:**
- `delivery_time_scaled`
- `review_polarity_scaled`
- `category_encoded`
- `helpfulness_ratio`
- `review_length`
- `is_high_rating`

📊 Feature importance is shown in the other tab.
""")
        st.markdown(helpfulness_note)

    elif model_option == "Random Forest":
        st.markdown("""
### 🌲 Random Forest
Random Forest is an ensemble of decision trees. Each tree is trained on a random part of the data. Predictions are made by combining all the trees' votes.

**Advantages:**
- Easy to use
- Works well with many features
- Reduces overfitting

**Features used:**
- `delivery_time`
- `review_polarity`
- `category_encoded`
- `Score`
- (You can also include `helpfulness_ratio` to enhance trust signal from user feedback.)
""")
        st.markdown(helpfulness_note)

    elif model_option == "Logistic Regression":
        st.markdown("""
### 📈 Logistic Regression
Logistic Regression models the probability of return using a weighted sum of the features, passed through a sigmoid function.

**Key Concepts:**
- Linear relationship between input and log-odds
- Fast training
- Easy to interpret weights

**Features used:**
- `delivery_time`
- `review_polarity`
- `category_encoded`
- `Score`
- (Optionally, include `helpfulness_ratio` for richer input signal.)

💡 Want to understand how each feature affects the outcome? Print `model.coef_` to see the weights!
""")
        st.markdown(helpfulness_note)
