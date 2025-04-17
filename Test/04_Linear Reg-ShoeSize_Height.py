# Simple linear regression : shoe size / height
# 64-42 Module HEVS

# First install libraries:
# pip install pandas seaborn scipy statsmodels

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from scipy import stats

# Set styles for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("viridis")

# 1. Load data with correct column names
df = pd.read_csv("Shoes_size_height.csv")
print("Dataframe overview:")
print(df.head())
print("\nColumn information:")
print(df.info())

# Verify that we have the expected columns
if 'x_Height' in df.columns and 'y_ShoeSize' in df.columns:
    print("\nColumns found correctly!")
else:
    print("\nWARNING: The expected column names ('x_Height', 'y_ShoeSize') were not found.")
    print("Available columns:", df.columns.tolist())

# 2. Missing value analysis
print("\n" + "="*60)
print(" "*15 + "MISSING VALUE ANALYSIS")
print("="*60)
print(df.isna().sum())
print(f"Percentage of missing values per column:")
print((df.isna().sum() / len(df) * 100).round(2))

# 3. Handling missing values (if necessary)
df_clean = df.dropna()
print(f"\nNumber of observations after removing missing values: {len(df_clean)}")

# 4. Descriptive statistics
print("\n" + "="*60)
print(" "*20 + "DESCRIPTIVE STATISTICS")
print("="*60)
print(df_clean.describe().round(4))

# 5. Data visualization
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='x_Height', y='y_ShoeSize', alpha=0.7, s=60)
plt.title('Relationship between Height and Shoe Size', fontsize=16, fontweight='bold')
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Shoe Size', fontsize=12)
plt.tight_layout()
plt.savefig('scatterplot_shoesize.png', dpi=300)

# 6. Correlation calculation
correlation = df_clean['x_Height'].corr(df_clean['y_ShoeSize'])
print("\n" + "="*60)
print(" "*20 + "CORRELATION ANALYSIS")
print("="*60)
print(f"Correlation coefficient (r) between height and shoe size: {correlation:.4f}")
print(f"Coefficient of determination (r²) (theoretical): {correlation**2:.4f}")

# 7. Linear regression
X_train = df_clean[['x_Height']].values
y_train = df_clean[['y_ShoeSize']].values

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Get the coefficients
slope = model.coef_[0][0]
intercept = model.intercept_[0]

# Make predictions
y_pred = model.predict(X_train)

# Calculate metrics
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_train, y_pred)

# 8. Display regression results
print("\n" + "="*60)
print(" "*15 + "REGRESSION MODEL RESULTS")
print("="*60)
print(f"✓ Equation of the line:              Shoe Size = {slope:.4f} × Height + {intercept:.4f}")
print(f"✓ Coefficient (slope):                {slope:.4f}")
print(f"✓ Intercept:               {intercept:.4f}")
print(f"✓ Mean Squared Error (MSE):   {mse:.4f}")
print(f"✓ Root MSE (RMSE):               {rmse:.4f}")
print(f"✓ Coefficient of determination (R²):  {r2:.4f}")
print("="*60)

# 9. Visualize the regression
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_clean, x='x_Height', y='y_ShoeSize', alpha=0.7, s=60, label='Data')
plt.plot(X_train, y_pred, color='red', linewidth=3, label='Regression')

# Equation of the line
equation = f'Shoe Size = {slope:.4f} × Height + {intercept:.4f}'
plt.annotate(equation, xy=(0.05, 0.95), xycoords='axes fraction', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

# R² on the graph
r2_text = f'R² = {r2:.4f}'
plt.annotate(r2_text, xy=(0.05, 0.89), xycoords='axes fraction', 
             fontsize=12, bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8))

plt.title('Linear Regression: Height vs Shoe Size', fontsize=16, fontweight='bold')
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Shoe Size', fontsize=12)
plt.legend(loc='upper left', fontsize=12)
plt.tight_layout()
plt.savefig('regression_shoesize.png', dpi=300)

# 10. Residual analysis
residuals = y_train - y_pred
df_clean['residuals'] = residuals

# Plot residuals
plt.figure(figsize=(12, 10))

# Residuals vs predicted values
plt.subplot(2, 2, 1)
sns.scatterplot(x=y_pred.flatten(), y=residuals.flatten())
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Predicted Shoe Sizes', fontsize=14)
plt.xlabel('Predicted Shoe Sizes', fontsize=12)
plt.ylabel('Residuals', fontsize=12)

# Distribution of residuals
plt.subplot(2, 2, 2)
sns.histplot(residuals.flatten(), kde=True)
plt.title('Distribution of Residuals', fontsize=14)
plt.xlabel('Residuals', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

# QQ-plot to check normality of residuals
plt.subplot(2, 2, 3)
stats.probplot(residuals.flatten(), plot=plt)
plt.title('QQ-Plot of Residuals', fontsize=14)

# Residuals vs independent variable
plt.subplot(2, 2, 4)
sns.scatterplot(x=X_train.flatten(), y=residuals.flatten())
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Residuals vs Height', fontsize=14)
plt.xlabel('Height (cm)', fontsize=12)
plt.ylabel('Residuals', fontsize=12)

plt.tight_layout()
plt.savefig('residual_analysis_shoesize.png', dpi=300)

# 11. Test for normality of residuals (Shapiro-Wilk test)
shapiro_test = stats.shapiro(residuals.flatten())
print("\n" + "="*60)
print(" "*15 + "RESIDUAL NORMALITY TEST")
print("="*60)
print(f"Shapiro-Wilk test: statistic={shapiro_test.statistic:.4f}, p-value={shapiro_test.pvalue:.4f}")
if shapiro_test.pvalue > 0.05:
    print("Residuals follow a normal distribution (p > 0.05)")
else:
    print("Residuals do not follow a normal distribution (p ≤ 0.05)")
print("="*60)

# 12. Test for homoscedasticity (constancy of residual variance)
from statsmodels.stats.diagnostic import het_breuschpagan
import statsmodels.api as sm

# For the Breusch-Pagan test, we need to add a constant to our variable X
X_with_const = sm.add_constant(X_train)
try:
    bp_test = het_breuschpagan(residuals.flatten(), X_with_const)
    
    print("\n" + "="*60)
    print(" "*15 + "HOMOSCEDASTICITY TEST")
    print("="*60)
    print(f"Breusch-Pagan test: statistic={bp_test[0]:.4f}, p-value={bp_test[1]:.4f}")
    if bp_test[1] > 0.05:
        print("The variance of residuals is constant (p > 0.05)")
    else:
        print("The variance of residuals is not constant (p ≤ 0.05)")
except Exception as e:
    print("\n" + "="*60)
    print(" "*15 + "HOMOSCEDASTICITY TEST")
    print("="*60)
    print(f"The Breusch-Pagan test could not be performed: {str(e)}")
    
    # Alternative: Goldfeld-Quandt test
    from statsmodels.stats.diagnostic import het_goldfeldquandt
    try:
        gq_test = het_goldfeldquandt(residuals.flatten(), X_train.flatten())
        print(f"Alternative Goldfeld-Quandt test: statistic={gq_test[0]:.4f}, p-value={gq_test[1]:.4f}")
        if gq_test[1] > 0.05:
            print("The variance of residuals is constant (p > 0.05)")
        else:
            print("The variance of residuals is not constant (p ≤ 0.05)")
    except Exception as e2:
        print(f"The Goldfeld-Quandt test could not be performed: {str(e2)}")
        print("Visual analysis recommended: examine the 'Residuals vs Predicted Values' graph")
print("="*60)

print("\nComplete analysis done! Plots have been saved.")
