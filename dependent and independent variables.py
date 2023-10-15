# Import necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression

# Generate some example data
# Replace this with your own data
independent_variable = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)  # Independent variable (e.g., X)
dependent_variable = np.array([2, 4, 5, 4, 6])  # Dependent variable (e.g., Y)

# Create a Linear Regression model
model = LinearRegression()

# Fit the model to your data
model.fit(independent_variable, dependent_variable)

# Make predictions
predicted_values = model.predict(independent_variable)

# Output the model parameters
print("Intercept:", model.intercept_)
print("Coefficient:", model.coef_[0])

# R-squared value to measure the goodness of fit
r_squared = model.score(independent_variable, dependent_variable)
print("R-squared:", r_squared)
