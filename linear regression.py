import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Sample data (replace with your own dataset)
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable (feature)
y = np.array([2, 4, 5, 4, 5])  # Target variable

# Create and train a Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the data points and the regression line
plt.scatter(X, y, label='Data Points')
plt.plot(X, y_pred, color='red', label='Regression Line')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend()
plt.show()

# Coefficients and intercept of the regression line
slope = model.coef_[0]
intercept = model.intercept_
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")
