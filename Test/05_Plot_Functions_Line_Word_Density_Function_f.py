import numpy as np
import matplotlib.pyplot as plt

# Parameters
l = 200  # Article length in words

# Creating the linear function f(x) = x/l
def f(x):
    return x/l

# Creating an array of x values
x = np.linspace(0, 50, 100)  # x values from 0 to 50

# Plotting the line
plt.figure(figsize=(10, 6))
plt.plot(x, f(x), 'b-', label=f'f(x) = x/{l}')

# Improving the display
plt.xlabel('Number of misleading words (x)')
plt.ylabel('Word density f(x,l)')
plt.title(f'Word density function for l = {l}')
plt.grid(True)
plt.legend()

# Adding an explanation
plt.text(30, 0.05, f"Slope of the line = 1/{l} = {1/l:.5f}", 
         bbox=dict(facecolor='lightgray', alpha=0.5))

plt.xlim(0, 50)
plt.ylim(0, 0.3)

plt.show()