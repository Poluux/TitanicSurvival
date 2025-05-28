import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D

# Chargement des données
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

# Prétraitement
df_model = df.copy()

# Garder uniquement les colonnes nécessaires et supprimer les lignes avec des valeurs manquantes
df_model = df_model[["Sex", "Age", "Pclass", "Survived"]].dropna()

# Encoder 'Sex' : male = 0, female = 1
df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})

# Créer des tranches d'âge (tranches de 5 ans)
df_model["AgeGroup"] = (df_model["Age"] // 5).astype(int)

# Définir les variables X et y
X = df_model[["Sex", "Pclass", "AgeGroup"]]
y = df_model["Survived"]

# Séparation entraînement / test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modèle de régression logistique
model = LogisticRegression()
model.fit(X_train, y_train)

# Évaluation
y_pred = model.predict(X_test)
print("\n--- Régression logistique avec SEX + Pclass + AgeGroup ---")
print(classification_report(y_test, y_pred))

# === Visualisation 3D ===

# Fixer Sexe = 1 (femme). Mettre 0 pour visualiser les hommes.
fixed_sex = 1

# Créer une grille pour Pclass (1 à 3) et AgeGroup (0 à 16 => jusqu'à 80+ ans)
pclass_range = np.arange(1, 4)
age_group_range = np.arange(0, 17)

Pclass_grid, Age_grid = np.meshgrid(pclass_range, age_group_range)
Pclass_flat = Pclass_grid.ravel()
Age_flat = Age_grid.ravel()

# Créer les entrées avec Sexe fixé
X_grid = np.column_stack([np.full_like(Pclass_flat, fixed_sex), Pclass_flat, Age_flat])

# Prédire la probabilité de survie
probas = model.predict_proba(X_grid)[:, 1]
Proba_grid = probas.reshape(Pclass_grid.shape)

# Graphique 3D
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Pclass_grid, Age_grid * 5, Proba_grid, cmap="coolwarm", edgecolor='k', alpha=0.8)

ax.set_xlabel("Classe")
ax.set_ylabel("Âge (années)")
ax.set_zlabel("Probabilité de survie")
ax.set_title("Probabilité de survie selon la classe et l'âge (Sexe = Femme)")
plt.tight_layout()
plt.show()
