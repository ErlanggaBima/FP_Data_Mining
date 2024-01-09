import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.cluster import MeanShift, estimate_bandwidth
import matplotlib.pyplot as plt

# Load the data
url = 'https://drive.google.com/file/d/1Y2YV23TOlgKzBBC6C4Hkkg-MHnzIuU8O/view?usp=sharing'
path = 'https://drive.google.com/uc?export=download&id='+url.split('/')[-2]
df = pd.read_csv(path)

# Streamlit app
st.title("Clustering and Analysis App")

# Display the DataFrame
st.subheader("Original DataFrame:")
st.dataframe(df)

# K-Means Clustering
numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
X = df[numeric_columns]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
n_clusters = 3
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Display K-Means results
st.subheader("K-Means Clustering Results:")
st.dataframe(df)

# Encode 'WHO Region' column
le = LabelEncoder()
df['WHO Region Encoded'] = le.fit_transform(df['WHO Region'])

# MeanShift Clustering
numeric_columns_ms = df.select_dtypes(include=['float64', 'int64']).columns
X_ms = df[numeric_columns_ms].to_numpy()
bandwidth = estimate_bandwidth(X_ms, quantile=0.2, n_samples=100)
if bandwidth < 1e-10:
    bandwidth = 1e-10
ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
ms.fit(X_ms)
labels_ms = ms.labels_
cluster_centers_ms = ms.cluster_centers_

# Display MeanShift results
st.subheader("MeanShift Clustering Results:")
st.write("Number of estimated clusters:", len(np.unique(labels_ms)))
st.write("Labels:", labels_ms)
st.write("Cluster Centers:", cluster_centers_ms)

# Plot Elbow method for K-Means
st.subheader("Elbow Method for K-Means:")
inertias = []
max_clusters = 10
for i in range(1, min(len(df), max_clusters) + 1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(X)
    inertias.append(kmeans.inertia_)

# Visualization of the Elbow method
fig, ax = plt.subplots()
ax.plot(range(1, min(len(df), max_clusters) + 1), inertias, marker='o')
ax.set_title('Elbow method')
ax.set_xlabel('Number of clusters')
ax.set_ylabel('Inertia')
st.pyplot(fig)

# Display K-Means clusters for selected number of clusters
num_clusters_selected = st.slider("Select the number of clusters for K-Means:", 1, 5, 3)
kmeans_selected = KMeans(n_clusters=num_clusters_selected)
kmeans_selected.fit(X)

# Display cluster results
df_selected = df.copy()
df_selected['Selected Cluster'] = kmeans_selected.labels_
st.subheader(f"Results for {num_clusters_selected} clusters (K-Means):")
st.dataframe(df_selected[['Selected Cluster'] + list(X.columns)])

# Additional functionalities can be added based on user interactions and preferences.

