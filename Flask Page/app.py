import os
import logging
from flask import Flask, request, render_template, jsonify
from pycaret.classification import load_model, predict_model
import pandas as pd

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
# Use an environment variable with a default fallback
model_path = os.environ.get('MODEL_PATH', 'best_model')

# model_path = 'best_model'

# Load the model
try:
    model = load_model(model_path)
    app.logger.info(f"Model loaded successfully from {model_path}")
except Exception as e:
    app.logger.error(f"Failed to load model: {str(e)}")
    model = None

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    probabilities = None
    error = None
    if request.method == 'POST':
        try:
            # Get values from form
            input_data = pd.DataFrame([[
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]], columns=['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'])
           
            # Make prediction
            predictions = predict_model(model, data=input_data)
           
            # Get the prediction
            if 'prediction_label' in predictions.columns:
                prediction = predictions['prediction_label'].iloc[0]
            elif 'prediction_score' in predictions.columns:
                prediction = predictions['prediction_score'].iloc[0]
            else:
                prediction = predictions.iloc[0, -1]  # Assume last column is the prediction
            
            # Map numeric predictions to class names
            class_names = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
            prediction = class_names.get(prediction, prediction)
           
            # Try to get probabilities if available
            prob_columns = [col for col in predictions.columns if col.startswith('prediction_score_')]
            if prob_columns:
                probabilities = {
                    class_names[i]: predictions[f'prediction_score_{i}'].iloc[0]
                    for i in range(3) if f'prediction_score_{i}' in predictions.columns
                }
           
            app.logger.info(f"Prediction: {prediction}")
            app.logger.info(f"Probabilities: {probabilities}")
        except Exception as e:
            app.logger.error(f"Error during prediction: {str(e)}")
            error = str(e)
   
    return render_template('index.html', prediction=prediction, probabilities=probabilities, error=error)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8000))
    app.run(host='0.0.0.0', port=port)