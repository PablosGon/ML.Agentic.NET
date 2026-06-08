from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score

print("=== LINEAR REGRESSION WITH SCIKIT-LEARN ===")
print()

t0 = perf_counter()
df = pd.read_csv('C:\\repos\\ML.Agentic.NET\\models\\regression\\HousePrice.csv')
t1 = perf_counter()
load_time = t1 * 1000 - t0 * 1000
print(f"Data loaded in {load_time:.2f} ms")
    
train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train = train.drop('House_Price', axis=1)
y_train = train['House_Price']
X_test = test.drop('House_Price', axis=1)
y_test = test['House_Price']

t2 = perf_counter()
split_time = t2 * 1000 - t1 * 1000
print(f"Data split into train and test sets in {split_time:.2f} ms")

categorical_cols = ['Neighborhood_Quality']
numeric_cols = [col for col in X_train.columns if col not in categorical_cols]
preprocessor = ColumnTransformer([
    ('onehot', OneHotEncoder(), categorical_cols),
    ('num', StandardScaler(), numeric_cols)
])
X_train_scaled = preprocessor.fit_transform(X_train)
X_test_scaled = preprocessor.transform(X_test)
model = LinearRegression()
model.fit(X_train_scaled, y_train)

t3 = perf_counter()
training_time = t3 * 1000 - t2 * 1000
print(f"Model trained in {training_time:.2f} ms")

y_pred = model.predict(X_test_scaled)
rmse = root_mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

t4 = perf_counter()
evaluation_time = t4 * 1000 - t3 * 1000
print(f"Model evaluated in {evaluation_time:.2f} ms")

print("=== RESULTS ===")
total_time = t4 * 1000 - t0 * 1000
print(f"Total execution time: {total_time:.2f} ms")
print(f"Root Mean Squared Error: {rmse:.2f}")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R-squared: {r2:.2f}")