# Email Spam Classifier 📧

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) techniques. 

This project implements and compares two commonly used models for text classification: **Logistic Regression** and **Multinomial Naive Bayes**, allowing for a practical understanding of how different algorithms perform on word-based data.

## 🚀 Features
- Text vectorisation using:
  - TF-IDF
  - Count Vectorizer
- Multiple machine learning models:
  - Logistic Regression
  - Multinomial Naive Bayes
- Hyperparameter tuning using GridSearchCV
- Model evaluation metrics:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Confusion matrix visualisation

## 🧠 Machine Learning Workflow
1. Load and clean dataset
2. Remove duplicate messages
3. Convert text into numerical features
4. Split data into training and testing sets
5. Train model
6. Tune hyperparameters
7. Evaluate model performance
8. Compare models

## 📊 Dataset
This project uses a labelled dataset of SMS messages:

- **Spam** → unwanted / promotional messages  
- **Ham** → normal messages  

File: `spamhamdata.csv`

## 🛠️ Technologies Used
- Python
- pandas
- NumPy
- scikit-learn
- matplotlib


## 📈 Example Metrics
The model outputs:
- Accuracy
- Precision
- Recall
- F1 Score

and displays a confusion matrix for visual evaluation.

## ▶️ How to Run

1. Install dependencies:
pip install pandas numpy scikit-learn matplotlib


2. Run the program:

3. Choose a model:
- `l` → Logistic Regression
- `n` → Naive Bayes

4. Optionally view evaluation results

## 🎯 Project Purpose

This project was built to:
- understand the machine learning pipeline
- work with text data
- compare different ML models
- practise hyperparameter tuning
