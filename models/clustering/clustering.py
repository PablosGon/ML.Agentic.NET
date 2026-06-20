from time import perf_counter

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import davies_bouldin_score
import joblib

print("=== K-MEANS WITH SCIKIT-LEARN ===")
print()

t0 = perf_counter()

df = pd.read_csv('C:\\repos\\ML.Agentic.NET\\models\\clustering\\online_retail_II.csv', 
                  sep=';', encoding='latin-1', decimal=',')
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'], format='%d/%m/%Y %H:%M', errors='coerce')
df['TotalAmount'] = df['Price'] * df['Quantity']
latest_date = df['InvoiceDate'].max()
rfm = df.groupby('Customer ID').agg({
    'InvoiceDate': lambda x: (latest_date - x.max()).days,
    'Invoice': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()
rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm = rfm.dropna()
rfm.drop("CustomerID", axis=1, inplace=True)
t1 = perf_counter()
load_time = (t1 - t0) * 1000
print(f"Data loaded in {load_time:.2f} ms")

X_train, X_test = train_test_split(rfm, test_size=0.2, random_state=42)
t2 = perf_counter()
split_time = (t2 - t1) * 1000
print(f"Data split into training and test sets in {split_time:.2f} ms")

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_train_scaled)
t3 = perf_counter()
train_time = (t3 - t2) * 1000
print(f"Model trained in {train_time:.2f} ms")

y_pred = kmeans.predict(X_test_scaled)
distances = kmeans.transform(X_test_scaled)
avg_distance = np.mean(np.min(distances, axis=1))
davies_bouldin = davies_bouldin_score(X_test_scaled, y_pred)
t4 = perf_counter()
eval_time = (t4 - t3) * 1000
print(f"Model evaluated in {eval_time:.2f} ms")

print()
print("-- RESULTS --")
total_time = (t4 - t0) * 1000
print(f"Total elapsed time: {total_time:.2f} ms")
print(f"Average distance: {avg_distance:.4f}")
print(f"Davies-Bouldin score: {davies_bouldin:.4f}")
print()

print("Saving model...")
joblib.dump(kmeans, 'C:\\repos\\ML.Agentic.NET\\models\\clustering\\CustomerClusteringModel.pkl')
print("Done.")

