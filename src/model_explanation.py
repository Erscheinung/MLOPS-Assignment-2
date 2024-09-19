from zenml import pipeline, step
from zenml.client import Client
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Any, Tuple, List
import joblib
import traceback

@step
def load_model_and_data() -> Tuple[Any, np.ndarray, list]:
    """Load the trained model and test data."""
    # Load the model
    model = joblib.load("models/best_model.pkl")
    
    # Load the test data
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    
    # Load feature names
    feature_names = np.load('data/feature_names.npy', allow_pickle=True).tolist()

    # debugging
    print(callable(model))
    
    return model, X_test, feature_names

class ModelWrapper:
    def __init__(self, model):
        self.model = model

    def __call__(self, X):
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)
    
def safe_index(arr, idx):
    if hasattr(arr, '__getitem__'):
        return arr[idx]
    return arr

@step
def explain_model(models: List, X_test: np.ndarray, feature_names: List) -> None:
    """Generate SHAP explanations for each model."""
    for i, model in enumerate(models):
        print(f"\nExplaining model {i}: {type(model).__name__}")
        try:
            # Create a SHAP explainer
            wrapped_model = ModelWrapper(model)
            
            if str(type(model).__name__) in ['RandomForestClassifier', 'ExtraTreesClassifier', 'GradientBoostingClassifier']:
                explainer = shap.TreeExplainer(model)
            else:
                n_background = min(100, X_test.shape[0])
                background_data = shap.kmeans(X_test, n_background)
                explainer = shap.KernelExplainer(wrapped_model, background_data)
            
            shap_values = explainer(X_test)

            print(f"Type of shap_values: {type(shap_values)}")
            print(f"Shape of shap_values: {shap_values.shape if hasattr(shap_values, 'shape') else 'No shape attribute'}")
            
            if isinstance(shap_values, list):
                print(f"Length of shap_values list: {len(shap_values)}")
                for j, sv in enumerate(shap_values):
                    print(f"  Shape of shap_values[{j}]: {sv.shape if hasattr(sv, 'shape') else 'No shape attribute'}")

            # Plot summary
            shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
            plt.tight_layout()
            plt.savefig(f"results/shap_summary_plot_model_{i}.png")
            plt.close()

            # Plot individual explanation for the first test instance
            first_instance = safe_index(shap_values, 0)
            if isinstance(first_instance, np.ndarray):
                shap.plots.waterfall(first_instance, show=False)
            else:
                print("Warning: Unable to generate waterfall plot due to unexpected SHAP value format.")
                shap.plots.beeswarm(shap_values, show=False)
            plt.tight_layout()
            plt.savefig(f"results/shap_individual_explanation_model_{i}.png")
            plt.close()

            print(f"SHAP explanations for model {i} generated and saved.")

            # Generate and print feature importance
            if hasattr(shap_values, 'values'):
                feature_importance = np.abs(shap_values.values).mean(0)
            elif isinstance(shap_values, list):
                feature_importance = np.abs(np.array(shap_values)).mean(0)
            else:
                feature_importance = np.abs(shap_values).mean(0)

            if feature_importance.ndim > 1:
                feature_importance = feature_importance.mean(1)

            importance_df = pd.DataFrame(list(zip(feature_names, feature_importance)), columns=['feature', 'importance'])
            importance_df = importance_df.sort_values('importance', ascending=False)
            print(f"\nFeature Importance for model {i} based on SHAP values:")
            print(importance_df)

        except Exception as e:
            print(f"An error occurred while explaining model {i}:")
            print(str(e))
            print("Traceback:")
            print(traceback.format_exc())
            print("Attempting alternative explanation method...")
            
            try:
                # Alternative: Use permutation importance
                from sklearn.inspection import permutation_importance
                r = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=0)
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': r.importances_mean
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                print("\nFeature Importance based on Permutation Importance:")
                print(importance_df)
                
                plt.figure(figsize=(10, 6))
                plt.bar(importance_df['feature'], importance_df['importance'])
                plt.title(f"Feature Importance for model {i}")
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f"results/feature_importance_model_{i}.png")
                plt.close()
                
            except Exception as e2:
                print(f"Alternative explanation method also failed for model {i}:")
                print(str(e2))
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