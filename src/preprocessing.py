import os
import sys

# Error debugging for running on conda
current_path = os.path.abspath(__file__)
root_path = os.path.dirname(os.path.dirname(current_path))
sys.path.append(root_path)

import pandas as pd
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from zenml import pipeline, step
from zenml.client import Client
from ydata_profiling import ProfileReport
from typing import Tuple

# Initialize ZenML
Client()

@step
def load_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the Iris dataset."""
    iris = load_iris()
    data = pd.DataFrame(iris.data, columns=iris.feature_names)
    target = pd.Series(iris.target, name='species')
    return data, target

@step
def perform_eda(data: pd.DataFrame, target: pd.Series) -> None:
    """Perform Exploratory Data Analysis."""
    df = pd.concat([data, target], axis=1)
    profile = ProfileReport(df, title="Iris Dataset Profiling Report", explorative=True)
    profile.to_file("results/iris_eda_report.html")
    print("EDA report generated: results/iris_eda_report.html")

@step
def preprocess_data(
    data: pd.DataFrame,
    target: pd.Series
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler, list]:
    """Preprocess the data, including splitting and scaling."""
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        data, target, test_size=0.2, random_state=42
    )
    
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the preprocessed data
    np.save('data/X_train.npy', X_train_scaled)
    np.save('data/X_test.npy', X_test_scaled)
    np.save('data/y_train.npy', y_train.values)
    np.save('data/y_test.npy', y_test.values)
    np.save('data/feature_names.npy', np.array(data.columns))
    
    print("Preprocessed data saved in 'data' directory")
    
    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler, data.columns.tolist()

@pipeline
def iris_preprocessing_pipeline():
    """Pipeline for preprocessing the Iris dataset."""
    data, target = load_data()
    perform_eda(data, target)
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(data, target)

def main():
    """Main function to run the preprocessing pipeline."""
    iris_preprocessing_pipeline()

if __name__ == "__main__":
    main()