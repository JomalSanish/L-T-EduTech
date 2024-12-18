import pandas as pd
import joblib
from sklearn.cluster import KMeans

# Load dataset
data_path = 'Mall_Customers.csv'  # Make sure the file exists at this location
customers = pd.read_csv(data_path)

# Select relevant features for clustering
X = customers[['Annual Income (k$)', 'Spending Score (1-100)']].values

# Train the KMeans model
kmeans = KMeans(n_clusters=5, random_state=42)  # 5 clusters for example
kmeans.fit(X)

# Save the trained KMeans model
model_path = 'kmeans_model.pkl'
joblib.dump(kmeans, model_path)

print(f"K-Means model trained and saved successfully at {model_path}")
