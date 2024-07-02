from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the trained model and scaler
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('scaler.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    transaction_amount = data['transaction_amount']
    transaction_time = data['transaction_time']
    user_age = data['user_age']
    user_transactions_last_24h = data['user_transactions_last_24h']
    hourly_transaction_rate = user_transactions_last_24h / 24

    features = np.array([[transaction_amount, transaction_time, user_age, user_transactions_last_24h, hourly_transaction_rate]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)

    return jsonify({'prediction': int(prediction[0])})

if __name__ == '__main__':
    app.run(debug=True)
