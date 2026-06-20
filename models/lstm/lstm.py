from time import perf_counter
from tensorflow import keras

print("=== LSTM WITH TensorFlow ===")
print()

t0 = perf_counter()
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000, seed=42)
t1 = perf_counter()
load_time = (t1 - t0) * 1000
print(f"Data loaded and split in {load_time:.0f} ms")

max_len = 256
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=max_len)
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=max_len)
t2 = perf_counter()
padding_time = (t2 - t1) * 1000
print(f"Data padded in {padding_time:.0f} ms")

model = keras.Sequential([
    keras.layers.Embedding(10000, 128, mask_zero=True),
    keras.layers.LSTM(128),
    keras.layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.fit(x_train, y_train, epochs=3, batch_size=64, validation_split=0.2)
t3 = perf_counter()
training_time = (t3 - t2) * 1000
print(f"Model trained in {training_time:.0f} ms")

metrics = model.evaluate(x_test, y_test, verbose=2)
t4 = perf_counter()
evaluation_time = (t4 - t3) * 1000
print(f"Model evaluated in {evaluation_time:.0f} ms")

print()
print("-- RESULTS --")
total_elapsed_time = load_time + padding_time + training_time + evaluation_time
print(f"Total elapsed time: {total_elapsed_time:.0f} ms")
print(f"Accuracy score: {metrics[1]}")