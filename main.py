#Import necessary libraries
import pandas as pd
import numpy as np
from typing import Dict, List, TypedDict
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

#Loading the dataset
df: pd.DataFrame = pd.read_csv("data/spamhamdata.csv", sep="\t", names=["label", "message"])
df.drop_duplicates(inplace=True)

X: pd.Series = df["message"]
y: pd.Series = df['label'].map({"spam": 1, "ham": 0})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#Defining a TypedDict (allows for more accurate type hints)
class ModelConfig(TypedDict):
    pipeline: Pipeline
    param_grid: Dict

#Creating a list of pipelines and param_grids
pipelines: List[ModelConfig] = [{
    'pipeline': Pipeline([
    ('vectoriser', TfidfVectorizer()),
    ('classifier', LogisticRegression())
]),
    'param_grid': {
         'classifier__max_iter'  : [100,1000,2500,5000],
        'classifier__C': np.logspace(-4,4,20)
    }
}, {
    'pipeline': Pipeline([
    ('vectoriser', CountVectorizer()),
    ('classifier', MultinomialNB())
]),
    'param_grid': {
    'classifier__alpha': [0.1, 0.5, 1.0, 2.0],
}
},]

#Function that trains the model
def model_training(pipeline_config: ModelConfig) -> str:
    output: str = "---Results----- \n"

    model: sklearn.base.BaseEstimator = pipeline_config["pipeline"]
    param_grid: Dict[str] = pipeline_config['param_grid']

    improved_model: GridSearchCV = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=True)

    improved_model.fit(X_train,y_train)

    output += f"best accuracy: {improved_model.best_score_} \n"
    output += f"best parameters: {improved_model.best_params_} \n"

    print(output)

    #User input code
    user_Input: str = input("Enter an email or SMS message to classify as spam or ham: ").strip()
    processed_Input: List[str] = [user_Input]
    if not user_Input:
        print("No input provided. Please enter a valid message.")
    else: 
        input_Prediction: np.array = improved_model.best_estimator_.predict(processed_Input)
        if 0 in input_Prediction:
            print("Prediction: Ham")
        else:
            print("Prediction: Spam")
    
    #Evaluation code
    evaluation_request: str = input("Would you like to see the evaluation of the model? (yes/no): ").lower().strip()

    if evaluation_request == "yes":

        y_pred: np.array = improved_model.best_estimator_.predict(X_test)

        accuracy: float = accuracy_score(y_test, y_pred)
        precision: float = precision_score(y_test, y_pred)
        recall: float = recall_score(y_test, y_pred)
        f1: float = f1_score(y_test, y_pred)

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1 score: {f1}')

        cm: np.array = confusion_matrix(y_test, y_pred)

        display: ConfusionMatrixDisplay = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=df.label.unique())
        display.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()
        return "Evaluation Complete."
    else:
        return "No evaluation."    

#Runs the main program
def main() -> None:
    model_choice: str = input("Select a model - Logistic Regression or Naive Bayes (l/n): ").lower().strip()

    if model_choice == "l":
        chosen_model = pipelines[0]
        print(model_training(chosen_model))

    elif model_choice == "n":
        chosen_model = pipelines[1]
        print(model_training(chosen_model))
    else:
        print("Please select an appropriate model (l or n).")

if __name__ == "__main__":
    main()