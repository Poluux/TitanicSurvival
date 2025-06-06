import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from mpl_toolkits.mplot3d import Axes3D
import joblib
from matplotlib import animation

# This file finds the regression and make 3D models to visualize it

# Loading data
df = pd.read_csv("../Project 2_Titanic-Dataset.csv")

# Preprocessing
df_model = df.copy()

# Keep only necessary columns and remove rows with missing values
df_model = df_model[["Sex", "Age", "Pclass", "Survived"]].dropna()

# Encode 'Sex': male = 0, female = 1
df_model["Sex"] = df_model["Sex"].map({"male": 0, "female": 1})

# Create age groups (5-year intervals)
df_model["AgeGroup"] = (df_model["Age"] // 5).astype(int)

# Define X and y variables
X = df_model[["Sex", "Pclass", "AgeGroup"]]
y = df_model["Survived"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Display the model equation
coef = model.coef_[0]
intercept = model.intercept_[0]
print("\nEquation of model (logit) :")
print(f"logit(p) = {intercept:.4f} + ({coef[0]:.4f} * Sex) + ({coef[1]:.4f} * Pclass) + ({coef[2]:.4f} * AgeGroup)")
print("Predicted probability : p = 1 / (1 + exp(-logit(p)))")

# Evaluation
y_pred = model.predict(X_test)
print("\n--- Logistic regression with SEX + Pclass + AgeGroup ---")
print(classification_report(y_test, y_pred))

# === 3D Visualization ===

# Fix Sex = 1 (female). Set to 0 to visualize for males.
fixed_sex = 1

# Create a grid for Pclass (1 to 3) and AgeGroup (0 to 16 => up to 80+ years)
pclass_range = np.arange(1, 4)
age_group_range = np.arange(0, 17)

Pclass_grid, Age_grid = np.meshgrid(pclass_range, age_group_range)
Pclass_flat = Pclass_grid.ravel()
Age_flat = Age_grid.ravel()

# Create input entries with fixed sex
X_grid = np.column_stack([np.full_like(Pclass_flat, fixed_sex), Pclass_flat, Age_flat])

# Convert to DataFrame with correct column names to avoid warning
X_grid_df = pd.DataFrame(X_grid, columns=["Sex", "Pclass", "AgeGroup"])

# Predict survival probability
probas = model.predict_proba(X_grid_df)[:, 1]
Proba_grid = probas.reshape(Pclass_grid.shape)

# 3D plot
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(Pclass_grid, Age_grid * 5, Proba_grid, cmap="coolwarm", edgecolor='k', alpha=0.8)

ax.set_xlabel("Class")
ax.set_ylabel("Age (year)")
ax.set_zlabel("Survival probability")
ax.set_title("Survival probablity according to age and class (Sex = Female)")
plt.tight_layout()

def rotate(angle):
    ax.view_init(elev=30, azim=angle)
 
# Create animation
ani = animation.FuncAnimation(fig, rotate, frames=np.arange(0, 360, 2), interval=100)
 
ani.save("rotation_survival.gif", writer="pillow", fps=20)
 
plt.show()