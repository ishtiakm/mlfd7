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

# Gradient descent function
def gradient_descent(initial_x, initial_y, learning_rate, iterations):
    x, y = initial_x, initial_y
    values = []  # To store the function value at each iteration
    for i in range(iterations):
        # Calculate the function value and store it
        values.append(f(x, y))
        
        # Update x and y using gradient descent
        grad = gradient(x, y)
        x -= learning_rate * grad[0]
        y -= learning_rate * grad[1]
    
    return values

# Parameters
initial_x = 0.1
initial_y = 0.1
iterations = 50

# Run gradient descent with learning rate 0.01
learning_rate_1 = 0.01
values_1 = gradient_descent(initial_x, initial_y, learning_rate_1, iterations)

# Run gradient descent with learning rate 0.1
learning_rate_2 = 0.1
values_2 = gradient_descent(initial_x, initial_y, learning_rate_2, iterations)

# Plot the results
plt.plot(range(iterations), values_1, label="Learning rate = 0.01")
plt.plot(range(iterations), values_2, label="Learning rate = 0.1")
plt.xlabel("Iteration")
plt.ylabel("Function Value")
plt.legend()
plt.title("Gradient Descent on f(x, y)")
plt.savefig("lr_conundrum.png")
plt.show()
