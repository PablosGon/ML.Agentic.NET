using static Tensorflow.KerasApi;

var ((xTrain, yTrain), (xTest, yTest)) = keras.datasets.imdb.load_data(num_words: 10000);

var layers = keras.layers;
var model = keras.Sequential(new List<Tensorflow.Keras.ILayer>
{
    layers.Embedding(10000, 128, mask_zero: true),
    layers.LSTM(128),
    layers.Dense(1, activation: "sigmoid")
});

model.compile(optimizer: "adam", loss: "binary_crossentropy", metrics: new[] { "accuracy" });

var modelFit = model.fit(xTrain, yTrain);

model.evaluate(yTrain, yTest);