import pandas as pd
import statsmodels.api as sm

# Création du DataFrame
data = {
    'Height': [196, 170, 173, 190, 193, 184, 190, 182, 187, 177, 178, 180, 179, 178, 165, 160, 168, 162, 175, 169, 168, 172, 168],
    'Gender': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Shoe_Size': [47, 43, 42, 45, 46, 43, 45, 43, 44, 42, 43, 42, 42, 40, 38, 37, 38, 37, 39, 38, 38, 39, 38]
}

df = pd.DataFrame(data)

# Ajout d'une constante à nos variables indépendantes
X = sm.add_constant(df[['Height', 'Gender']])

# Régression linéaire multiple
model = sm.OLS(df['Shoe_Size'], X).fit()

# Affichage des résultats
print(model.summary())