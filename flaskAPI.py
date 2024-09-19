from flask import Flask, request, jsonify
import pickle # we will use pickle to deploy the best model
import numpy as np

app = Flask(__name__)

# Load model
with open('/models/best_model.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).reshape(1,-1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)