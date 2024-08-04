from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load('../models/isolation_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    data = pd.read_csv(file)
    predictions = model.predict(data)
    return jsonify(predictions.tolist())

if __name__ == "__main__":
    app.run(port=5001)
