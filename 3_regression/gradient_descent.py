import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

# Load and prepare data
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")
df = df[["Sex", "Age", "Pclass", "Survived"]].dropna()
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df["AgeGroup"] = (df["Age"] // 5).astype(int)

X = df[["Sex", "Pclass", "AgeGroup"]].values
y = df["Survived"].values

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 1. Sklearn model ===
model = LogisticRegression()
model.fit(X_train, y_train)
sk_w0 = model.intercept_[0]
sk_w1, sk_w2, sk_w3 = model.coef_[0]

# === 2. Gradient descent (without normalization) ===
# Add bias (column of ones)
X_train_b = np.hstack([np.ones((X_train.shape[0], 1)), X_train])

# Initialize weights
weights = np.zeros(X_train_b.shape[1])
learning_rate = 0.01
epochs = 100000

# Sigmoid function
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient descent
for epoch in range(epochs):
    z = np.dot(X_train_b, weights)
    predictions = sigmoid(z)
    errors = predictions - y_train
    gradient = np.dot(X_train_b.T, errors) / len(y_train)
    weights -= learning_rate * gradient
    if epoch % 100 == 0:
        loss = -np.mean(y_train * np.log(predictions + 1e-9) + (1 - y_train) * np.log(1 - predictions + 1e-9))
        print(f"Epoch {epoch} - Loss: {loss:.4f}")

# Learned weights
gd_w0, gd_w1, gd_w2, gd_w3 = weights

# === Comparison ===
print("\n--- Coefficients sklearn ---")
print(f"w0 = {sk_w0:.4f}")
print(f"w1 = {sk_w1:.4f} (Sex)")
print(f"w2 = {sk_w2:.4f} (Pclass)")
print(f"w3 = {sk_w3:.4f} (AgeGroup)")

print("\n--- Coefficients gradient descent ---")
print(f"w0 = {gd_w0:.4f}")
print(f"w1 = {gd_w1:.4f} (Sex)")
print(f"w2 = {gd_w2:.4f} (Pclass)")
print(f"w3 = {gd_w3:.4f} (AgeGroup)")
