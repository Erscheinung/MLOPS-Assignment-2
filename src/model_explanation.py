from zenml import pipeline, step
from zenml.client import Client
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Any, Tuple
import joblib

@step
def load_model_and_data() -> Tuple[Any, np.ndarray, list]:
    """Load the trained model and test data."""
    # Load the model
    model = joblib.load("models/best_model.pkl")
    
    # Load the test data
    X_test = np.load('data/X_test.npy')
    
    # Load feature names
    feature_names = np.load('data/feature_names.npy').tolist()
    
    return model, X_test, feature_names

@step
def explain_model(model: Any, X_test: np.ndarray, feature_names: list) -> None:
    """Generate SHAP explanations for the model."""
    # Create a SHAP explainer
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)

    # Plot summary
    shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
    plt.tight_layout()
    plt.savefig("results/shap_summary_plot.png")
    plt.close()

    # Plot individual explanation for the first test instance
    shap.plots.waterfall(shap_values[0], show=False)
    plt.tight_layout()
    plt.savefig("results/shap_individual_explanation.png")
    plt.close()

    print("SHAP explanations generated and saved as 'results/shap_summary_plot.png' and 'results/shap_individual_explanation.png'")

    # Generate and print feature importance
    feature_importance = np.abs(shap_values.values).mean(0)
    importance_df = pd.DataFrame(list(zip(feature_names, feature_importance)), columns=['feature', 'importance'])
    importance_df = importance_df.sort_values('importance', ascending=False)
    print("\nFeature Importance based on SHAP values:")
    print(importance_df)

@pipeline
def iris_model_explanation_pipeline():
    """Pipeline for explaining the Iris classification model."""
    model, X_test, feature_names = load_model_and_data()
    explain_model(model, X_test, feature_names)
    
def main():
    """Main function to run the model explanation pipeline."""
    iris_model_explanation_pipeline()

if __name__ == "__main__":
    main()