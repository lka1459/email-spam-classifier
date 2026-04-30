# Email Spam Classifier 📧

A machine learning project that classifies SMS messages as **spam** or **ham (not spam)** using Natural Language Processing (NLP) techniques. 

This project implements and compares two commonly used models for text classification: **Logistic Regression** and **Multinomial Naive Bayes**, allowing for a practical understanding of how different algorithms perform on word-based data. 

## 🚀 Features
- Text vectorisation using:
  - TF-IDF
  - Count Vectorizer
- Machine learning models:
  - Logistic Regression
  - Multinomial Naive Bayes
- End-to-end ML pipelines (vectorisation + model)
- Hyperparameter tuning using GridSearchCV
- Model evaluation using:
  - Accuracy
  - Precision
  - Recall
  - F1 Score
- Confusion matrix visualisation
- Interactive user input for real-time message classification

## 🧠 Machine Learning Workflow
1. Load and clean dataset  
2. Remove duplicate messages  
3. Split data into training and testing sets  
4. Build pipelines (vectorisation + model)  
5. Train models using GridSearchCV  
6. Tune hyperparameters  
7. Evaluate model performance on unseen data  
8. Compare model results
9. Classify custom user input messages  

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

A confusion matrix is also generated for visual performance analysis.

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
