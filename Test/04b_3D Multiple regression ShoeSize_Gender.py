# 3D graphic of multiple linear regression : shoe size / height and gender
# 64-42 Module HEVS


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression

# Data creation "Gender"
data = pd.DataFrame({
    'Height': [196, 170, 173, 190, 193, 184, 190, 182, 187, 177, 178, 180, 179, 178, 165, 160, 168, 162, 175, 169, 168, 172, 168],
    'Gender': [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'Shoe_Size': [47,43,42,45,46,43,45,43,44,42,43,42,42,40,38,37,38,37,39,38,38,39,38]
})

# Multiple linear regression model training
model = LinearRegression()
X = data[['Height', 'Gender']]
y = data['Shoe_Size']
model.fit(X, y)

# Creating a scatter plot
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Display of coloured real dots by gender
scatter = ax.scatter(data['Height'], data['Gender'], data['Shoe_Size'],
                     c=data['Gender'], cmap='coolwarm', s=100,
                     label='Données réelles')

# Creating the prediction surface
x_grid = np.linspace(data['Height'].min(), data['Height'].max(), 20)
y_grid = np.array([0, 1])
X_grid, Y_grid = np.meshgrid(x_grid, y_grid)
Z_grid = model.predict(pd.DataFrame({'Height': X_grid.ravel(), 'Gender': Y_grid.ravel()})).reshape(X_grid.shape)

# Prediction surface display
surf = ax.plot_surface(X_grid, Y_grid, Z_grid, cmap='viridis', alpha=0.5)

# Paramétrage des axes et du titre
ax.set_xlabel('Height (cm)')
ax.set_ylabel('Gender')
ax.set_zlabel('Shoe Size (EU)')
ax.set_yticks([0, 1])
ax.set_yticklabels(['Female', 'Male'])
ax.set_title('3D Multiple Regression: Shoe Size prediction')

# Adding a colour bar
fig.colorbar(surf, shrink=0.5, aspect=5)

# Display of regression equation and R²
equation = f'Shoe Size = {model.intercept_:.2f} + {model.coef_[0]:.2f}*Height + {model.coef_[1]:.2f}*Gender'
r2 = f'R² = {model.score(X, y):.2f}'
ax.text2D(0.05, 0.95, equation, transform=ax.transAxes)
ax.text2D(0.05, 0.90, r2, transform=ax.transAxes)

plt.tight_layout()
plt.show()

print(equation)
print(r2)
