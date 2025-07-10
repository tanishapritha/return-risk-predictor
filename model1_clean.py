#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from textblob import TextBlob

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression


# In[2]:


from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

import joblib

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, plot_importance

sns.set(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)


# In[3]:


# from kaggle.api.kaggle_api_extended import KaggleApi
# import zipfile


# os.environ['KAGGLE_CONFIG_DIR'] = os.path.join(os.getcwd(), 'kaggle')
# api = KaggleApi()
# api.authenticate()


# api.dataset_download_files('arhamrumi/amazon-product-reviews', path='.', unzip=True)


# with zipfile.ZipFile('amazon-product-reviews.zip', 'r') as zip_ref:
#     zip_ref.extractall('amazon_reviews')


# In[4]:


df = pd.read_csv("amazon_reviews/Reviews.csv")
df.head()


# In[5]:


df.info()


# In[6]:


df.isnull().sum()


# In[7]:


df.describe(include='all')


# ### 🔍 Step 1: Dataset Overview
# 
# We begin by loading the raw Amazon product reviews dataset, which includes metadata such as:
# 
# - `ProductId`, `UserId`
# - `Score` (rating)
# - `Summary` and full `Text` of the review
# - Timestamps (in Unix format)
# 
# This dataset will form the foundation for our return risk prediction pipeline.

# In[8]:


sns.countplot(x='Score', data=df)
plt.title("Review Score Distribution")
plt.xlabel("Score")
plt.ylabel("Number of Reviews")
plt.show()


# In[9]:


def map_sentiment(score):
    if score >= 4:
        return "positive"
    elif score == 3:
        return "neutral"
    else:
        return "negative"

df['Sentiment'] = df['Score'].apply(map_sentiment)
df['Sentiment'].value_counts()


# Data Cleaning and Preprocessing

# In[ ]:


cols_to_keep = ['ProductId', 'UserId', 'HelpfulnessNumerator', 'HelpfulnessDenominator',
                'Score', 'Time', 'Summary', 'Text']
df = df[cols_to_keep]
df.dropna(subset=['Summary', 'Text'], inplace=True)

# Convert Unix time to datetime
df['review_date'] = pd.to_datetime(df['Time'], unit='s')

# Simulate delivery_time in days (1–10)
np.random.seed(42)
df['delivery_time'] = np.random.randint(1, 10, size=len(df))


df['category'] = df['ProductId'].astype(str).str[0]
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])


df['review_polarity'] = df['Text'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

# Normalize
scaler = StandardScaler()
df[['delivery_time_scaled', 'review_polarity_scaled']] = scaler.fit_transform(
    df[['delivery_time', 'review_polarity']]
)

df.head()


# In[13]:


np.random.seed(42)
df['was_returned'] = np.random.choice([0, 1], size=len(df), p=[0.8, 0.2])


plt.figure(figsize=(10, 6))
sns.heatmap(df[['delivery_time_scaled', 'review_polarity_scaled', 'category_encoded', 'Score', 'was_returned']].corr(), 
            annot=True, cmap='coolwarm', fmt='.2f')
plt.title("]Correlation Heatmap")
plt.show()

# Distribution of Review Polarity
plt.figure(figsize=(8, 5))
sns.histplot(df['review_polarity'], bins=50, kde=True, color='purple')
plt.title("]Review Polarity Distribution")
plt.xlabel("Polarity Score (-1 to +1)")
plt.ylabel("Frequency")
plt.show()

# Return Rate by Product Category
plt.figure(figsize=(10, 6))
category_return = df.groupby('category')['was_returned'].mean().sort_values()
sns.barplot(x=category_return.index, y=category_return.values, palette="Set2")
plt.title("]Average Return Rate by Category")
plt.xlabel("Category (from ProductId prefix)")
plt.ylabel("Return Rate")
plt.show()

# Return Rate by Delivery Time
plt.figure(figsize=(10, 6))
delivery_return = df.groupby('delivery_time')['was_returned'].mean().sort_index()
sns.barplot(x=delivery_return.index, y=delivery_return.values, palette="coolwarm")
plt.title("]Return Rate vs. Delivery Time")
plt.xlabel("Delivery Time (days)")
plt.ylabel("Return Rate")
plt.show()


# Logistic Regression

# In[14]:


# Baseline features (no scaled values)
features_lr = ['delivery_time', 'review_polarity', 'category_encoded', 'Score']
X_lr = df[features_lr]
y = ((df['Score'] <= 2) | (df['review_polarity'] < 0)).astype(int)  # Baseline target

X_train_lr, X_test_lr, y_train_lr, y_test_lr = train_test_split(X_lr, y, test_size=0.2, random_state=42)


# In[15]:


lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train_lr, y_train_lr)

y_pred_lr = lr_model.predict(X_test_lr)

print("Logistic Regression")
print("Accuracy:", accuracy_score(y_test_lr, y_pred_lr))
print("AUC Score:", roc_auc_score(y_test_lr, lr_model.predict_proba(X_test_lr)[:, 1]))
print(classification_report(y_test_lr, y_pred_lr, zero_division=0))


# Random Forest Classifier

# In[16]:


rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_lr, y_train_lr)

y_pred_rf = rf_model.predict(X_test_lr)

print("Random Forest")
print("Accuracy:", accuracy_score(y_test_lr, y_pred_rf))
print("AUC Score:", roc_auc_score(y_test_lr, rf_model.predict_proba(X_test_lr)[:, 1]))
print(classification_report(y_test_lr, y_pred_rf, zero_division=0))


# In[17]:


cm_rf = confusion_matrix(y_test_lr, y_pred_rf)
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix – Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()


# XGBoost

# In[20]:


# Feature Scaling

# Add additional features
df['helpfulness_ratio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, np.nan)
df['helpfulness_ratio'].fillna(0, inplace=True)
df['review_length'] = df['Text'].apply(lambda x: len(str(x)))
df['is_high_rating'] = (df['Score'] >= 4).astype(int)

# Scale required features
scaler = StandardScaler()
df[['delivery_time_scaled', 'review_polarity_scaled']] = scaler.fit_transform(df[['delivery_time', 'review_polarity']])


# In[21]:


# Add new features
df['helpfulness_ratio'] = df['HelpfulnessNumerator'] / df['HelpfulnessDenominator'].replace(0, np.nan)
df['helpfulness_ratio'] = df['helpfulness_ratio'].fillna(0)

df['review_length'] = df['Text'].apply(lambda x: len(str(x)))
df['is_high_rating'] = (df['Score'] >= 4).astype(int)

# Feature scaling for delivery_time and review_polarity
scaler = StandardScaler()
df[['delivery_time_scaled', 'review_polarity_scaled']] = scaler.fit_transform(
    df[['delivery_time', 'review_polarity']]
)


# In[22]:


final_features = [
    'delivery_time_scaled',
    'review_polarity_scaled',
    'category_encoded',
    'helpfulness_ratio',
    'review_length',
    'is_high_rating'
]

X = df[final_features]
y = ((df['Score'] <= 2) | (df['review_polarity'] < 0)).astype(int)  # Binary target: was_returned

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Handle class imbalance
scale = len(y_train[y_train == 0]) / len(y_train[y_train == 1])
print(f"scale_pos_weight = {scale:.2f}")


# In[23]:


xgb_model = XGBClassifier(
    scale_pos_weight=scale,
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

xgb_model.fit(X_train, y_train)


# In[24]:


y_pred = xgb_model.predict(X_test)
y_proba = xgb_model.predict_proba(X_test)[:, 1]

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


# joblib

# In[25]:


import joblib
joblib.dump(xgb_model, "return_risk_xgboost_model.pkl")

# Save the scaler too
joblib.dump(scaler, "scaler.pkl")

print("Model and scaler saved successfully!")


# In[26]:


plot_importance(xgb_model, importance_type='gain')


# In[ ]:




