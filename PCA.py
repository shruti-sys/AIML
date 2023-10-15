import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Generate some sample data (replace this with your own dataset)
np.random.seed(0)
n_samples = 100
n_features = 2
X = np.random.randn(n_samples, n_features)

# Create a PCA model
n_components = 1  # Number of components (desired dimensionality)
pca = PCA(n_components=n_components)

# Fit the model and transform the data
X_pca = pca.fit_transform(X)

# Variance explained by each principal component
explained_variance_ratio = pca.explained_variance_ratio_

print(f"Explained Variance Ratio: {explained_variance_ratio}")

# Plot the original data and the PCA-transformed data
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1])
plt.title("Original Data")

plt.subplot(1, 2, 2)
plt.scatter(X_pca, np.zeros_like(X_pca))
plt.title("PCA-Transformed Data")

plt.show()
