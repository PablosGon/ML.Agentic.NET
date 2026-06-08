from time import perf_counter

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

print("=== NAIVE BAYES WITH SCIKIT-LEARN ===")
print()

t0 = perf_counter()
df = pd.read_csv('C:\\repos\\ML.Agentic.NET\\models\\naivebayes\\Phishing_Email.csv', 
                  engine='python', on_bad_lines='skip', dtype={'Email Text': str, 'Email Type': str})
t1 = perf_counter()
load_time = t1 * 1000 - t0 * 1000
print(f"Data loaded in {load_time:.2f} ms")

df = df.drop(columns=['Unnamed: 0'], errors='ignore')
df = df.dropna(subset=['Email Text', 'Email Type'])
df = df[df['Email Type'].str.strip().ne('')]
df['Email Type'] = df['Email Type'].str.strip()
df['Email Text'] = df['Email Text'].str.strip()

class_counts = df['Email Type'].value_counts()
valid_classes = class_counts[class_counts >= 2].index
df = df[df['Email Type'].isin(valid_classes)]
print(f"Total samples after cleanup: {len(df)}")
print(f"Classes: {df['Email Type'].unique()}")



train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['Email Type'])
X_train = train['Email Text']
y_train = train['Email Type']
X_test = test['Email Text']
y_test = test['Email Type']

t2 = perf_counter()
split_time = (t2 - t1) * 1000
print(f"Data split into train and test sets in {split_time:.2f} ms")

pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english', ngram_range=(1, 2), max_features=5000)),
    ('clf', MultinomialNB()),
])

t3 = perf_counter()
pipeline.fit(X_train, y_train)
t4 = perf_counter()
train_time = (t4 - t3) * 1000
print(f"Model trained in {train_time:.2f} ms")

predictions = pipeline.predict(X_test)

eval_time = (perf_counter() - t4) * 1000
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy:.4f}")
print()
print("Classification report:")
print(classification_report(y_test, predictions, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_test, predictions))
print(f"Evaluation finished in {eval_time:.2f} ms")
