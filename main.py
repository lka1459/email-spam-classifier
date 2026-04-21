import pandas as pd
import numpy as np
from typing import Dict, List, TypedDict, Sequence, Any
import sklearn
from scipy.sparse import spmatrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, ConfusionMatrixDisplay
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

df: pd.DataFrame = pd.read_csv("spamhamdata.csv", sep="\t", names=["label", "message"])
df.drop_duplicates(inplace=True)

class ModelConfig(TypedDict):
    model: sklearn.base.BaseEstimator
    vectoriser: sklearn.base.BaseEstimator
    param_grid: Dict[str, Sequence[Any]]


parameters_list: List[ModelConfig] = [{
    'model': LogisticRegression(max_iter=200),
    'vectoriser': TfidfVectorizer(stop_words='english'),
    'param_grid': {
         'max_iter'  : [100,1000,2500,5000],
        'C': np.logspace(-4,4,20)
    }
}, {
    'model': MultinomialNB(),
    'vectoriser': CountVectorizer(stop_words='english'),
    'param_grid': {
    'alpha': [0.1, 0.5, 1.0, 2.0],
}
}]

def model_training(chosen_model: Dict) -> str:
    output: str = "---Results----- \n"
    X: spmatrix = chosen_model['vectoriser'].fit_transform(df['message'])
    y: pd.Series = df['label'].map({"spam": 1, "ham": 0}) 

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    model: sklearn.base.BaseEstimator = chosen_model["model"]
    model.fit(X_train, y_train)

    param_grid: Dict[str] = chosen_model['param_grid']

    improved_model: GridSearchCV = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, verbose=True)

    improved_model.fit(X_train,y_train)

    output += f"best accuracy: {improved_model.best_score_} \n"
    output += f"best parameters: {improved_model.best_params_} \n"

    print(output)

    evaluation_request: str = input("Would you like to see the evaluation of the model? (yes/no): ").lower()

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

def main() -> None:
    model_choice: str = input("Select a model - Logistic Regression or Naive Bayers (l/n): ").lower()

    if model_choice == "l":
        chosen_model = parameters_list[0]
        print(model_training(chosen_model))

    elif model_choice == "n":
        chosen_model = parameters_list[1]
        print(model_training(chosen_model))
    else:
        print("Please select an appropriate model (l or n).")

if __name__ == "__main__":
    main()