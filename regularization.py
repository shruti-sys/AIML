import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create Ridge (L2) and Lasso (L1) regression models
ridge_model = Ridge(alpha=0.1)  # Ridge regularization
lasso_model = Lasso(alpha=0.1)  # Lasso regularization

# Train the Ridge model
ridge_model.fit(X_train, y_train)
y_pred_ridge = ridge_model.predict(X_test)

# Train the Lasso model
lasso_model.fit(X_train, y_train)
y_pred_lasso = lasso_model.predict(X_test)

# Calculate mean squared error for both models
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)

# Plot the data and regression lines
plt.scatter(X, y, label='Data')
plt.plot(X_test, y_pred_ridge, color='red', label='Ridge Regression')
plt.plot(X_test, y_pred_lasso, color='green', label='Lasso Regression')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Ridge and Lasso Regularization')
plt.legend()
plt.show()

print(f"Ridge Regression MSE: {mse_ridge:.4f}")
print(f"Lasso Regression MSE: {mse_lasso:.4f}")
