# Return Risk Predictor

[Check out the live app (deployed on Hugging Face!)](https://huggingface.co/spaces/tanishapritha/return-risk-predictor)

A machine learning-powered Streamlit app that predicts the probability of a product being returned based on customer reviews, delivery metadata, and review ratings.

## Overview

Product returns in e-commerce lead to significant losses. This project applies machine learning and natural language processing to predict the return likelihood of a product using customer reviews, metadata, and delivery data.

Built with:
- Python and Streamlit for the UI
- Scikit-learn and XGBoost for machine learning
- TextBlob for sentiment analysis
- Plotly and Seaborn for visualization

## Features

| Feature               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Sentiment Analysis    | Uses TextBlob to analyze the tone of customer reviews                       |
| Delivery Time Impact  | Evaluates how delivery duration affects return chances                      |
| Rating Integration    | Leverages 1–5 # Return Risk Predictor

A machine learning-powered Streamlit app that predicts the probability of a product being returned based on customer reviews, delivery metadata, and review ratings.

## Overview

Product returns in e-commerce lead to significant losses. This project applies machine learning and natural language processing to predict the return likelihood of a product using customer reviews, metadata, and delivery data.

Built with:
- Python and Streamlit for the UI
- Scikit-learn and XGBoost for machine learning
- TextBlob for sentiment analysis
- Plotly and Seaborn for visualization

## Features

| Feature               | Description                                                                 |
|-----------------------|-----------------------------------------------------------------------------|
| Sentiment Analysis    | Uses TextBlob to analyze the tone of customer reviews                       |
| Delivery Time Impact  | Evaluates how delivery duration affects return chances                      |
| Rating Integration    | Leverages 1–5 star ratings to gauge satisfaction                            |
| Helpfulness Ratio     | Measures how helpful other users found the review                           |
| Category Encoding     | Simulated product category derived from ProductId                           |
| Multiple ML Models    | Compare predictions using Logistic Regression, Random Forest, and XGBoost   |
| Model Insights        | Learn how each model works and what features it relies on                   |

## Dataset Used

The project uses the Amazon Product Reviews dataset from Kaggle. Core columns include:

| Column                   | Description                                 |
|--------------------------|---------------------------------------------|
| `Text`                   | Full customer review                        |
| `Score`                  | Star rating (1 to 5)                        |
| `HelpfulnessNumerator`   | Number of users who found it helpful        |
| `HelpfulnessDenominator` | Total users who voted                       |
| `ProductId`, `UserId`    | Product and user identifiers                |
| `Time`                   | Review timestamp (Unix format)              |

Additional engineered features:
- `delivery_time`: Simulated shipping duration
- `category_encoded`: Encoded first character of ProductId
- `review_polarity`: Sentiment score using TextBlob
- `review_length`: Character count of the review
- `helpfulness_ratio`: Calculated as  
  `helpfulness_ratio = HelpfulnessNumerator / HelpfulnessDenominator`  
  (set to 0 when denominator is 0)

## Models Compared

| Model               | Advantages                                       | Use Case                                |
|---------------------|--------------------------------------------------|------------------------------------------|
| Logistic Regression | Fast and interpretable                          | Baseline modeling                        |
| Random Forest       | Handles nonlinear relationships, less overfit   | General tabular problems                 |
| XGBoost             | High accuracy, scalable, feature-aware          | Preferred for structured feature data    |

## Performance

| Model               | Accuracy | AUC Score |
|---------------------|----------|-----------|
| Logistic Regression | 0.79     | 0.72      |
| Random Forest       | 0.84     | 0.81      |
| XGBoost             | 0.87     | 0.89      |

## 📽 Demo

![App Demo](demo.gif)


## Installation

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the app

```bash
streamlit run app.py
```

## Deploying to Hugging Face (Basic Guide)

1. Create a new Space at [https://huggingface.co/spaces](https://huggingface.co/spaces) and select **Streamlit** as the SDK.

2. Upload the following files:

```
app.py  
requirements.txt  
lr_model.pkl  
rf_model.pkl  
xg_model.pkl  
scaler.pkl  
features.txt  
demo.gif (optional)
```

3. At the top of your `README.md`, add the following configuration block:

```yaml
---
title: Return Risk Predictor
sdk: streamlit
app_file: app.py
---
```


Hugging Face: https://huggingface.co/tanishapritha

If you found this project useful, feel free to star the repository or share it.
star ratings to gauge satisfaction                            |
| Helpfulness Ratio     | Measures how helpful other users found the review                           |
| Category Encoding     | Simulated product category derived from ProductId                           |
| Multiple ML Models    | Compare predictions using Logistic Regression, Random Forest, and XGBoost   |
| Model Insights        | Learn how each model works and what features it relies on                   |

## Dataset Used

The project uses the Amazon Product Reviews dataset from Kaggle. Core columns include:

| Column                   | Description                                 |
|--------------------------|---------------------------------------------|
| `Text`                   | Full customer review                        |
| `Score`                  | Star rating (1 to 5)                        |
| `HelpfulnessNumerator`   | Number of users who found it helpful        |
| `HelpfulnessDenominator` | Total users who voted                       |
| `ProductId`, `UserId`    | Product and user identifiers                |
| `Time`                   | Review timestamp (Unix format)              |

Additional engineered features:
- `delivery_time`: Simulated shipping duration
- `category_encoded`: Encoded first character of ProductId
- `review_polarity`: Sentiment score using TextBlob
- `review_length`: Character count of the review
- `helpfulness_ratio`: Calculated as  
  `helpfulness_ratio = HelpfulnessNumerator / HelpfulnessDenominator`  
  (set to 0 when denominator is 0)

## Models Compared

| Model               | Advantages                                       | Use Case                                |
|---------------------|--------------------------------------------------|------------------------------------------|
| Logistic Regression | Fast and interpretable                          | Baseline modeling                        |
| Random Forest       | Handles nonlinear relationships, less overfit   | General tabular problems                 |
| XGBoost             | High accuracy, scalable, feature-aware          | Preferred for structured feature data    |

## Performance

| Model               | Accuracy | AUC Score |
|---------------------|----------|-----------|
| Logistic Regression | 0.79     | 0.72      |
| Random Forest       | 0.84     | 0.81      |
| XGBoost             | 0.87     | 0.89      |

## Running Locally

1. Clone the repository

```bash
git clone https://github.com/your-username/return-risk-predictor.git
cd return-risk-predictor
