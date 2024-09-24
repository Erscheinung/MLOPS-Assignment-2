from zenml import pipeline, step
from pycaret.classification import load_model, get_config
import shap
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from typing import Any, Tuple, List
from sklearn.inspection import permutation_importance

@step
def load_model_and_data() -> Tuple[Any, pd.DataFrame, List[str], pd.Series]:
    """Load the trained model and test data from .npy files."""
    model = load_model("models/best_model")
    X_test = np.load('data/X_test.npy', allow_pickle=True)
    y_test = np.load('data/y_test.npy', allow_pickle=True)
    feature_names = np.load('data/feature_names.npy', allow_pickle=True)
    
    # Convert numpy arrays to pandas DataFrame and Series
    X_test = pd.DataFrame(X_test, columns=feature_names)
    y_test = pd.Series(y_test, name='target')
    
    print(f"Model type: {type(model)}")
    print(f"X_test shape: {X_test.shape}")
    print(f"Feature names: {feature_names}")
    
    return model, X_test, feature_names.tolist(), y_test

@step
def explain_model(model: Any, X_test: pd.DataFrame, feature_names: List[str], y_test: pd.Series) -> None:
    print(f"\nExplaining model: {type(model).__name__}")
    
    # Extract the final estimator from the PyCaret pipeline
    final_estimator = model.steps[-1][1]
    
    # Try SHAP explanation
    try:
        # Attempt to create a SHAP explainer
        if hasattr(final_estimator, "predict_proba"):
            explainer = shap.TreeExplainer(final_estimator)
        else:
            explainer = shap.KernelExplainer(final_estimator.predict, X_test.iloc[:100])
        
        shap_values = explainer.shap_values(X_test)

        # Plot the SHAP summary plot
        plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_test, plot_type="bar", feature_names=feature_names, show=False)
        plt.title("SHAP Feature Importance")
        plt.tight_layout()
        plt.savefig("results/shap_summary_plot.png")
        plt.close()
        print("SHAP summary plot saved to results/shap_summary_plot.png")

        # Print feature importance
        if isinstance(shap_values, list):
            # Multi-class case
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(np.array(shap_values)).mean(0).mean(0)
            })
        else:
            # Binary classification case
            feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': np.abs(shap_values).mean(0)
            })
        
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        print("\nFeature Importance based on SHAP values:")
        print(feature_importance)

        # Generate SHAP force plot for a single prediction
        plt.figure(figsize=(20, 3))
        shap.force_plot(explainer.expected_value[0] if isinstance(explainer.expected_value, list) else explainer.expected_value, 
                        shap_values[0] if isinstance(shap_values, list) else shap_values[0,:], 
                        X_test.iloc[0,:], feature_names=feature_names, matplotlib=True, show=False)
        plt.title("SHAP Force Plot for Single Prediction")
        plt.tight_layout()
        plt.savefig("results/shap_force_plot.png", bbox_inches='tight')
        plt.close()
        print("SHAP force plot saved to results/shap_force_plot.png")
        
    except Exception as e:
        print(f"Error in SHAP explanation: {str(e)}")
        print("Falling back to permutation feature importance...")
        
        # Use permutation feature importance
        perm_importance = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
        
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': perm_importance.importances_mean
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        
        print("\nFeature Importance based on Permutation Importance:")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        feature_importance.plot(x='feature', y='importance', kind='bar')
        plt.title("Feature Importance (Permutation)")
        plt.tight_layout()
        plt.savefig("results/permutation_importance_plot.png")
        plt.close()
        print("Permutation importance plot saved to results/permutation_importance_plot.png")

@pipeline
def iris_model_explanation_pipeline():
    """Pipeline for explaining the Iris classification model."""
    model, X_test, feature_names, y_test = load_model_and_data()
    explain_model(model, X_test, feature_names, y_test)

if __name__ == "__main__":
    iris_model_explanation_pipeline()