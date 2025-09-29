import numpy as np
import pickle
from sklearn.metrics import accuracy_score

# 1. Load the model and scaler
with open('output/model.pkl', 'rb') as f:
    data = pickle.load(f)
model, scaler = data['model'], data['scaler']

# 2. Load features and labels
X = np.load('output/features.npy')
y = np.load('output/labels.npy')

# 3. Scale features (if not already scaled)
X_scaled = scaler.transform(X)

# 4. Make predictions
y_pred = model.predict(X_scaled)

# 5. Calculate accuracy
accuracy = accuracy_score(y, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")