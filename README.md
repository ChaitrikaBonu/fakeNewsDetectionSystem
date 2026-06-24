# 📰 Fake News Detection System

## Overview

The Fake News Detection System is a Machine Learning project that classifies news articles as Real or Fake using Natural Language Processing (NLP) techniques. The model is trained on a dataset containing real and fake news articles and can predict the authenticity of user-provided news content.

## Features

- Detects Fake and Real News
- NLP-based Text Processing
- TF-IDF Feature Extraction
- Logistic Regression Classification
- Confidence Score Display
- Interactive Streamlit Web Interface
- Fast and Easy Predictions

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- Pickle

## Dataset

Dataset Used:
- Fake.csv
- True.csv

Source:
https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset

## Project Workflow

1. Data Collection
2. Data Preprocessing
3. Data Cleaning
4. Label Encoding
5. TF-IDF Vectorization
6. Train-Test Split
7. Logistic Regression Training
8. Model Evaluation
9. Model Saving
10. Streamlit Deployment

## Project Structure

FakeNewsDetection/

├── app.py

├── fake_news_model.pkl

├── vectorizer.pkl

├── Fake.csv

├── True.csv

├── requirements.txt

└── README.md

## Installation

### Clone Repository

```bash
git clone <repository-url>
cd FakeNewsDetection
