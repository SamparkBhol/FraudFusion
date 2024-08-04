from flask import Flask, request, jsonify
from src.models.anomaly_detection import load_model
import pandas as pd

app = Flask(__name__)

@app.route('/')
def index():
    return "Fraud Detection API"

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)
    model = load_model('../models/isolation_forest_model.pkl')
    predictions = model.predict(data)
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    app.run(debug=True)
