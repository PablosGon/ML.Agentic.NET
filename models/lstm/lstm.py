from time import perf_counter
import tensorflow as tf
from tensorflow import keras

print("=== LSTM WITH TensorFlow ===")
print()

start_time = perf_counter()
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000, seed=42)
load_time = (perf_counter() - start_time) * 1000
print(f"Data loaded and split in {load_time:.0f} ms")

max_len = 256
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)

start_time = perf_counter()
model = keras.Sequential([
    keras.layers.Embedding(10000, 128, mask_zero=True),
    keras.layers.LSTM(128),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
training_time = (perf_counter() - start_time) * 1000
print(f"Model trained in {training_time:.0f} ms")

start_time = perf_counter()
metrics = model.evaluate(x_test, y_test, verbose=2)
evaluation_time = (perf_counter() - start_time) * 1000
print(f"Model evaluated in {evaluation_time:.0f} ms")

print()
print("-- RESULTS --")
total_elapsed_time = load_time + training_time + evaluation_time
print(f"Total elapsed time: {total_elapsed_time:.0f} ms")
print(f"Evaluation metrics: {metrics}")