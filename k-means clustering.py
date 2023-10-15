import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs

# Generate some random data for clustering
n_samples = 300
n_features = 2
n_clusters = 3
X, _ = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_clusters, random_state=42)

# Create a K-Means clustering model
kmeans = KMeans(n_clusters=n_clusters, random_state=0)

# Fit the model to your data
kmeans.fit(X)

# Get the cluster assignments for each data point
labels = kmeans.labels_

# Get the cluster centroids
centroids = kmeans.cluster_centers_

# Plot the data points and cluster centroids
plt.scatter(X[:, 0], X[:, 1], c=labels)
plt.scatter(centroids[:, 0], centroids[:, 1,], marker='x', s=200, linewidths=3, color='r')
plt.title('K-Means Clustering')
plt.show()
