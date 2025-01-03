import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load the dataset (Replace 'your_data.csv' with your actual file path)
data = pd.read_csv('random_customer_data.csv')

# Display basic info about the dataset
print("Initial Data Info:")
print(data.info())

# Basic preprocessing (Handle missing values if any)
data = data.dropna()

# Select relevant features (Example: 'Age', 'Annual Income', 'Spending Score')
features = ['Age', 'Annual Income', 'Spending Score']
X = data[features]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Determine the optimal number of clusters using the Elbow Method
inertia = []
K = range(1, 11)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plot the Elbow Curve
plt.figure(figsize=(10, 6))
plt.plot(K, inertia, 'bx-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k')
plt.show()

# From the Elbow Curve, choose the optimal number of clusters (e.g., k=4)
k_optimal = 4
kmeans = KMeans(n_clusters=k_optimal, random_state=42)
kmeans.fit(X_scaled)

# Add cluster labels to the original dataset
data['Cluster'] = kmeans.labels_

# Calculate silhouette score
silhouette_avg = silhouette_score(X_scaled, kmeans.labels_)
print(f"Silhouette Score for k={k_optimal}: {silhouette_avg}")

# Visualize the clusters (2D plot for simplicity)
plt.figure(figsize=(10, 6))
sns.scatterplot(
    x=X_scaled[:, 1], y=X_scaled[:, 2], hue=data['Cluster'], palette='Set2', s=100
)
plt.title('Customer Segments Based on Clusters')
plt.xlabel('Annual Income (Standardized)')
plt.ylabel('Spending Score (Standardized)')
plt.legend(title='Cluster')
plt.show()

# Summary Statistics for Each Cluster
cluster_summary = data.groupby('Cluster')[features].mean()
print("Cluster Summary:")
print(cluster_summary)

# Visualize the distribution of clusters
plt.figure(figsize=(10, 6))
sns.countplot(x='Cluster', data=data, palette='Set2')
plt.title('Distribution of Customers Across Clusters')
plt.xlabel('Cluster')
plt.ylabel('Number of Customers')
plt.show()

# Save clustered data to a new CSV file
data.to_csv('segmented_customers.csv', index=False)
