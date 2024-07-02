import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

# Step 1: Generate Synthetic Data
def generate_synthetic_data(num_samples=10000):
    np.random.seed(42)
    data = {
        'transaction_amount': np.random.uniform(1, 10000, num_samples),
        'transaction_time': np.random.randint(0, 24, num_samples),
        'user_age': np.random.randint(18, 70, num_samples),
        'user_transactions_last_24h': np.random.randint(0, 50, num_samples),
        'is_fraud': np.random.choice([0, 1], num_samples, p=[0.98, 0.02])  # 2% fraud rate
    }
    return pd.DataFrame(data)

data = generate_synthetic_data()

# Step 2: Preprocess Data
data['hourly_transaction_rate'] = data['user_transactions_last_24h'] / 24

# Splitting the data
X = data.drop('is_fraud', axis=1)
y = data['is_fraud']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Scaling the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Step 3: Train the Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 4: Save the Model and Scaler
joblib.dump(model, 'fraud_detection_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler have been saved successfully.")
