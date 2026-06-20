from time import perf_counter

import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== NAIVE BAYES WITH SCIKIT-LEARN ===")
print()

t0 = perf_counter()

csv.field_size_limit(10_000_000)
df = pd.read_csv(r'C:\repos\ML.Agentic.NET\models\naivebayes\Phishing_Email.csv')
df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df.dropna(subset=['Email Text', 'Email Type'])
t1 = perf_counter()
load_time = t1 * 1000 - t0 * 1000
print(f"Data loaded in {load_time:.2f} ms")

train, test = train_test_split(df, test_size=0.2, random_state=42)
X_train = train['Email Text']
y_train = train['Email Type']
X_test = test['Email Text']
y_test = test['Email Type']
t2 = perf_counter()
split_time = (t2 - t1) * 1000
print(f"Data split into train and test sets in {split_time:.2f} ms")

vectorizer = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)
model = MultinomialNB()
model.fit(X_train_vectorized, y_train)
t3 = perf_counter()
train_time = (t3 - t2) * 1000
print(f"Model trained in {train_time:.2f} ms")

predictions = model.predict(X_test_vectorized)
accuracy = accuracy_score(y_test, predictions)
report = classification_report(y_test, predictions, digits=4)
conf_matrix = confusion_matrix(y_test, predictions)
t4 = perf_counter()
eval_time = (t4 - t3) * 1000
print(f"Evaluation finished in {eval_time:.2f} ms")

print("-- RESULTS --")
print(f"Total elapsed time: {(t4 - t0) * 1000:.2f} ms")
print(f"Accuracy: {accuracy:.4f}")
print()
print("Classification report:")
print(report)
print("Confusion matrix:")
print(conf_matrix)
