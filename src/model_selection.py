from zenml import pipeline, step
import pandas as pd
import numpy as np
from pycaret.classification import setup, compare_models, pull, finalize_model, save_model, predict_model
from sklearn.metrics import accuracy_score, classification_report
from typing import Tuple, Any

@step
def load_preprocessed_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Load the preprocessed data from the previous pipeline."""
    # Loading from data directory, saved earlier from the preprocessing pipeline
    X_train = np.load('data/X_train.npy')
    X_test = np.load('data/X_test.npy')
    y_train = np.load('data/y_train.npy')
    y_test = np.load('data/y_test.npy')
    feature_names = np.load('data/feature_names.npy', allow_pickle=True)
   
    # Convert to pandas DataFrame and Series
    X_train = pd.DataFrame(X_train, columns=feature_names)
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_train = pd.Series(y_train, name='species')
    y_test = pd.Series(y_test, name='species')
   
    return X_train, X_test, y_train, y_test

@step
def automl_model_selection(
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Tuple[Any, pd.DataFrame]:
    """Perform AutoML model selection using PyCaret."""
    # Combine features and target
    data = X_train.copy()
    data['target'] = y_train
   
    # Initialize PyCaret setup
    clf = setup(data=data, target='target', use_gpu=True)
   
    # Compare models and select the best one
    best_model = compare_models(n_select=1)
   
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
    X_test: pd.DataFrame,
    y_test: pd.Series
) -> None:
    """Evaluate the best model on the test set."""
    predictions = predict_model(model, data=X_test)
    y_pred = predictions['prediction_label'].values if 'prediction_label' in predictions.columns else predictions['prediction_score'].values
   
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
   
    print(f"Model Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # Save evaluation results
    with open("results/model_evaluation.txt", "w") as f:
        f.write(f"Model Accuracy: {accuracy}\n\n")
        f.write("Classification Report:\n")
        f.write(report)

@pipeline
def iris_model_selection_pipeline():
    """Pipeline for model selection and evaluation on the Iris dataset."""
    X_train, X_test, y_train, y_test = load_preprocessed_data()
    best_model, model_results = automl_model_selection(X_train, y_train)
    evaluate_model(best_model, X_test, y_test)

    
def main():
    """Main function to run the model selection pipeline."""
    iris_model_selection_pipeline()

if __name__ == "__main__":
    main()