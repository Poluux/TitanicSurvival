import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# This file train the model to find a regression and saves it, so we can use it in other files

# Load data
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

# Preprocessing
df_model = df[["Sex", "Age", "Pclass", "Survived"]].dropna()
df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})
df_model["AgeGroup"] = (df_model["Age"] // 5).astype(int)

# Variables
X = df_model[["Sex", "Pclass", "AgeGroup"]]
y = df_model["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LogisticRegression()
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, "logistic_model.joblib")

# Save X_test and y_test for evaluation
np.save("X_test.npy", X_test.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

print("Model and test data saved.")
