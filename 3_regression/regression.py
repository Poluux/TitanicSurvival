from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler

# Chargement des données
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

# Prétraitement
df_model = df.copy()

# Encodage de 'Sex' en numérique
df_model['Sex'] = df_model['Sex'].map({'male': 0, 'female': 1})

# Supprimer les lignes avec valeurs manquantes sur les colonnes utilisées
df_model = df_model.dropna(subset=['Age', 'Fare', 'Sex', 'Pclass', 'SibSp', 'Parch', 'Survived'])

# Définir X et y
X = df_model[['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']]
y = df_model['Survived']

# Séparer en données d'entraînement et test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Affichage des probabilités de survie
proba_survival = model.predict_proba(X_test)[:, 1]
print("Exemple de probabilités de survie :")
print(proba_survival[:10])

# Régression logistique basée uniquement sur l'âge

# Supprimer les lignes avec valeurs manquantes dans Age et Survived
df_age_model = df_model.dropna(subset=["Age", "Survived"])

# Redéfinir X et y pour ce modèle
X_age = df_age_model[["Age"]]
y_age = df_age_model["Survived"]

# Séparer en entraînement/test
X_train_age, X_test_age, y_train_age, y_test_age = train_test_split(X_age, y_age, test_size=0.2, random_state=42)

# Créer et entraîner le modèle
age_model = LogisticRegression()
age_model.fit(X_train_age, y_train_age)

# Prédictions et rapport de classification
y_pred_age = age_model.predict(X_test_age)
print("\n--- Régression logistique avec AGE uniquement ---")
print(classification_report(y_test_age, y_pred_age))

# Affichage des probabilités de survie en fonction de l'âge
proba_age = age_model.predict_proba(X_test_age)[:, 1]
print("Exemple de probabilités de survie (basé uniquement sur l'âge) :")
for age_val, prob in zip(X_test_age["Age"].head(10), proba_age[:10]):
    print(f"Âge: {age_val:.1f} ➤ Probabilité de survie: {prob:.3f}")

# Créer une plage d'âges pour la courbe
ages = np.linspace(df_age_model["Age"].min(), df_age_model["Age"].max(), 300).reshape(-1, 1)
probas = age_model.predict_proba(ages)[:, 1]

# Tracer la courbe logistique
plt.figure(figsize=(10, 6))
plt.plot(ages, probas, color="blue", label="Probabilité prédite de survie")
plt.scatter(X_test_age["Age"], y_test_age, alpha=0.3, color="red", label="Données réelles (survie)")
plt.xlabel("Âge")
plt.ylabel("Probabilité de survie")
plt.title("Régression logistique : âge vs probabilité de survie")
plt.legend()
plt.grid(True)
plt.show()