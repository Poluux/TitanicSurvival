import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pandas as pd

# Chargement des données
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

# Copier et encoder la variable 'Sex' en numérique
df_model = df.copy()
df_model['Sex'] = df_model['Sex'].map({'male': 0, 'female': 1})

# Supprimer les lignes avec valeurs manquantes dans les colonnes utilisées
cols = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Survived']
df_model = df_model.dropna(subset=cols)

variables = ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch']
target = 'Survived'

print("Corrélation entre chaque variable et Survived :\n")
for var in variables:
    corr = df_model[var].corr(df_model[target])
    print(f"{var}: Corrélation = {corr:.3f}")