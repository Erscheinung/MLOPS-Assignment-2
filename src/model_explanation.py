from zenml import pipeline, step
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
from typing import Any, Tuple, List
from pycaret.classification import load_model
import traceback

@step
def load_model_and_data() -> Tuple[Any, pd.DataFrame, list, pd.Series]:
    """Load the trained model and test data."""
    try:
        # Load the model
        model = load_model("models/best_model")  # Use PyCaret's load_model function
        
        # Load the test data
        X_test = np.load('data/X_test.npy', allow_pickle=True)
        y_test = np.load('data/y_test.npy', allow_pickle=True)
        feature_names = np.load('data/feature_names.npy', allow_pickle=True)
        
        # Convert to pandas DataFrame and Series
        X_test = pd.DataFrame(X_test, columns=feature_names)
        y_test = pd.Series(y_test, name='species')

        print(f"Model type: {type(model)}")
        print(f"Is model callable: {callable(model)}")
        print(f"X_test shape: {X_test.shape}")
        print(f"X_test columns: {X_test.columns.tolist()}")
        
        if model is None:
            raise ValueError("Loaded model is None")
        
        return model, X_test, feature_names.tolist(), y_test
    except Exception as e:
        print(f"Error in load_model_and_data: {str(e)}")
        raise

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
def explain_model(model: Any, X_test: pd.DataFrame, feature_names: List, y_test: pd.Series) -> None:
    """Generate SHAP explanations for the model."""
    print(f"\nExplaining model: {type(model).__name__}")
    try:
        # For PyCaret models, we need to use the 'estimator_' attribute
        if hasattr(model, 'estimator_'):
            model_to_explain = model.estimator_
        else:
            model_to_explain = model

        # Create a SHAP explainer
        if hasattr(model_to_explain, 'predict_proba'):
            explainer = shap.KernelExplainer(model_to_explain.predict_proba, X_test.iloc[:100])
        else:
            explainer = shap.KernelExplainer(model_to_explain.predict, X_test.iloc[:100])
        
        shap_values = explainer.shap_values(X_test)

        # Plot summary
        shap.summary_plot(shap_values, X_test, feature_names=feature_names, show=False)
        plt.tight_layout()
        plt.savefig("results/shap_summary_plot.png")
        plt.close()

        # Plot individual explanation for the first test instance
        if isinstance(shap_values, list):
            shap_values_plot = shap_values[0]
        else:
            shap_values_plot = shap_values
        shap.plots.waterfall(shap_values_plot[0], show=False)
        plt.tight_layout()
        plt.savefig("results/shap_individual_explanation.png")
        plt.close()

        print("SHAP explanations generated and saved.")

        # Generate and print feature importance
        if isinstance(shap_values, list):
            feature_importance = np.abs(np.array(shap_values)).mean(0).mean(0)
        else:
            feature_importance = np.abs(shap_values).mean(0)
        importance_df = pd.DataFrame(list(zip(feature_names, feature_importance)), columns=['feature', 'importance'])
        importance_df = importance_df.sort_values('importance', ascending=False)
        print("\nFeature Importance based on SHAP values:")
        print(importance_df)

    except Exception as e:
        print("An error occurred while explaining the model:")
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
            plt.title("Feature Importance")
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig("results/feature_importance.png")
            plt.close()
            
        except Exception as e2:
            print("Alternative explanation method also failed:")
            print(str(e2))

@pipeline
def iris_model_explanation_pipeline():
    """Pipeline for explaining the Iris classification model."""
    model, X_test, feature_names, y_test = load_model_and_data()
    explain_model(model, X_test, feature_names, y_test)
    
def main():
    """Main function to run the model explanation pipeline."""
    iris_model_explanation_pipeline()

if __name__ == "__main__":
    main()