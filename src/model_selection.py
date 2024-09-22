from zenml import pipeline, step
from zenml.client import Client
import pandas as pd
import numpy as np
from pycaret.classification import setup, save_model, compare_models, pull, predict_model
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

def automl_model_selection(
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Tuple[Any, pd.DataFrame]:
    """Perform AutoML model selection using PyCaret."""
    # Prepare the data for PyCaret
    data = pd.DataFrame(X_train, columns=[f'feature_{i}' for i in range(X_train.shape[1])])
    data['target'] = y_train
   
    # Initialize PyCaret setup
    clf = setup(data=data, target='target', use_gpu=True)
   
    # Compare models and select the best one
    best_model = compare_models(n_select=1)  # Changed from 5 to 1
   
    # Get the results
    model_results = pull()
   
    # Save the results
    model_results.to_csv("results/model_comparison_results.csv")
    print("Model comparison results saved to: results/model_comparison_results.csv")
   
    # Finalize the model
    final_model = finalize_model(best_model)
   
    # Save the best model
    save_model(final_model, "models/best_model")
    print("Best model saved to: models/best_model.pkl")
   
    return final_model, model_results

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
    # evaluate_model(best_model, X_test, y_test)


    
def main():
    """Main function to run the model selection pipeline."""
    iris_model_selection_pipeline()

if __name__ == "__main__":
    main()