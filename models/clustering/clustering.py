from time import perf_counter
from datetime import datetime

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib

print("=== K-MEANS WITH SCIKIT-LEARN ===")
print()

t0 = perf_counter()
df = pd.read_csv('C:\\repos\\ML.Agentic.NET\\models\\clustering\\online_retail_II.csv', 
                  sep=';', encoding='latin-1', on_bad_lines='skip', decimal=',')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')
df = df.dropna(subset=['InvoiceDate', 'Customer ID', 'Price', 'Quantity'])
df['TotalAmount'] = df['Price'] * df['Quantity']

t1 = perf_counter()
load_time = (t1 - t0) * 1000
print(f"Data loaded in {load_time:.2f} ms")

t2 = perf_counter()
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm.dropna()

t3 = perf_counter()
rfm_time = (t3 - t2) * 1000
print(f"RFM calculated in {rfm_time:.2f} ms")

t4 = perf_counter()
X_train, X_test = train_test_split(rfm[['Recency', 'Frequency', 'Monetary']], 
                                     test_size=0.2, random_state=42)
t5 = perf_counter()
split_time = (t5 - t4) * 1000
print(f"Data split into training and test sets in {split_time:.2f} ms")

t6 = perf_counter()
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
kmeans.fit(X_train_scaled)

t7 = perf_counter()
train_time = (t7 - t6) * 1000
print(f"Model trained in {train_time:.2f} ms")

t8 = perf_counter()
y_pred = kmeans.predict(X_test_scaled)
distances = kmeans.transform(X_test_scaled)
avg_distance = np.mean(np.min(distances, axis=1))
silhouette = silhouette_score(X_test_scaled, y_pred)
t9 = perf_counter()
eval_time = (t9 - t8) * 1000
print(f"Model evaluated in {eval_time:.2f} ms")

print()
print("-- RESULTS --")
total_time = (t9 - t0) * 1000
print(f"Total elapsed time: {total_time:.2f} ms")
print(f"Average distance: {avg_distance:.4f}")
print(f"Silhouette score: {silhouette:.4f}")
print()

print("Saving model...")
joblib.dump(kmeans, 'C:\\repos\\ML.Agentic.NET\\models\\clustering\\CustomerClusteringModel.pkl')
print("Done.")

