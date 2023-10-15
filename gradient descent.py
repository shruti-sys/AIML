import numpy as np

# Generate some sample data
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# Initialize model parameters
theta0 = np.random.randn()
theta1 = np.random.randn()

# Define hyperparameters
learning_rate = 0.01
n_iterations = 1000

# Perform gradient descent
for iteration in range(n_iterations):
    # Compute predictions
    y_pred = theta0 + theta1 * X
    
    # Compute the gradient of the cost function with respect to theta0 and theta1
    gradient_theta0 = (1 / len(X)) * np.sum(y_pred - y)
    gradient_theta1 = (1 / len(X)) * np.sum((y_pred - y) * X)
    
    # Update model parameters using the gradient and learning rate
    theta0 = theta0 - learning_rate * gradient_theta0
    theta1 = theta1 - learning_rate * gradient_theta1

# Print the final model parameters
print(f"Theta0: {theta0:.2f}")
print(f"Theta1: {theta1:.2f}")
