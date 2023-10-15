import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt

# Generate sample data (replace with your own dataset)
X = np.sort(5 * np.random.rand(80, 1), axis=0)
y = np.sin(X).ravel()

# Add noise to the data
y[::5] += 3 * (0.5 - np.random.rand(16))

# Create SVR model
svr_rbf = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.2)

# Fit the model to the data
svr_rbf.fit(X, y)

# Predict values for a range of data points
X_test = np.linspace(0, 5, 100)[:, np.newaxis]
y_rbf = svr_rbf.predict(X_test)

# Plot the results
plt.scatter(X, y, color='darkorange', label='Data')
plt.plot(X_test, y_rbf, color='navy', lw=2, label='RBF Model')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Support Vector Regression')
plt.legend()
plt.show()
