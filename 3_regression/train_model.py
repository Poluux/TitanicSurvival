import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

# Chargement des données
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

# Prétraitement
df_model = df[["Sex", "Age", "Pclass", "Survived"]].dropna()
df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})
df_model["AgeGroup"] = (df_model["Age"] // 5).astype(int)

# Variables
X = df_model[["Sex", "Pclass", "AgeGroup"]]
y = df_model["Survived"]

# Séparation train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Entraînement modèle
model = LogisticRegression()
model.fit(X_train, y_train)

# Sauvegarder le modèle
joblib.dump(model, "logistic_model.joblib")

# Sauvegarder X_test et y_test pour l'évaluation
np.save("X_test.npy", X_test.to_numpy())
np.save("y_test.npy", y_test.to_numpy())

print("Model and test data saved.")
