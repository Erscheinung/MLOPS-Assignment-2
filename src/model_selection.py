from zenml import pipeline, step
from zenml.client import Client
import pandas as pd
import numpy as np
from pycaret.classification import setup, compare_models, pull, predict_model
from sklearn.metrics import accuracy_score, classification_report
import joblib
from typing import Tuple, Any

@step
def load_preprocessed_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the preprocessed data from the previous pipeline."""
    # Loading from data directory, saved earlier from the preprocessing pipeline
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    
    return X_train, X_test, y_train, y_test

@step
def automl_model_selection(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple[Any, pd.DataFrame]:
    """Perform AutoML model selection using PyCaret."""
    # Prepare the data for PyCaret
    data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    data['target'] = y_train
    
    # Initialize PyCaret setup
    clf = setup(data=data, target='target', silent=True, use_gpu=True)
    
    # Compare models
    best_model = compare_models(n_select=5)
    
    # Get the results
    model_results = pull()
    
    return best_model, model_results

@step
def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> None:
    """Evaluate the best model on the test set."""
    y_pred = predict_model(model, data=pd.DataFrame(X_test))['prediction'].values
    
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

@pipeline
def iris_model_selection_pipeline():
    """Pipeline for model selection and evaluation on the Iris dataset."""
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    best_model, model_results = automl_model_selection(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)
    
    # Save the results
    model_results.to_csv("results/model_comparison_results.csv")
    print("Model comparison results saved to: results/model_comparison_results.csv")
    
    # Save the best model
    joblib.dump(best_model, "models/best_model.pkl")
    print("Best model saved to: models/best_model.pkl")

if __name__ == "__main__":
    iris_model_selection_pipeline()