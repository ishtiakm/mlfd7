# Function to run gradient descent and return final position and function value
import numpy as np
import matplotlib.pyplot as plt

# Define the function f(x, y)
def f(x, y):
    return x**2 + 2*y**2 + 2 * np.sin(2 * np.pi * x) * np.sin(2 * np.pi * y)

# Define the gradient of f(x, y)
def gradient(x, y):
    df_dx = 2 * x + 4 * np.pi * np.cos(2 * np.pi * x) * np.sin(2 * np.pi * y)
    df_dy = 4 * y + 4 * np.pi * np.sin(2 * np.pi * x) * np.cos(2 * np.pi * y)
    return np.array([df_dx, df_dy])
    
def gradient_descent_with_output(initial_x, initial_y, learning_rate, iterations):
    x, y = initial_x, initial_y
    for _ in range(iterations):
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
    return x, y, f(x, y)

# Initial points
initial_points = [
    (0.1, 0.1),
    (1, 1),
    (-0.5, -0.5),
    (-1, -1)
]

# Parameters
learning_rate = 0.01
iterations = 50

# Table to store results
results = []

# Run gradient descent for each initial point
for initial_x, initial_y in initial_points:
    final_x, final_y, min_value = gradient_descent_with_output(initial_x, initial_y, learning_rate, iterations)
    results.append((initial_x, initial_y, final_x, final_y, min_value))

# Print the results as a table
import pandas as pd
df_results = pd.DataFrame(results, columns=["Initial x", "Initial y", "Final x", "Final y", "Minimum Value"])
print(df_results)