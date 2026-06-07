using System.Diagnostics;
using Tensorflow.Contexts;
using Tensorflow.Keras.Metrics;
using static Tensorflow.KerasApi;

var clock = new Stopwatch();

Console.WriteLine("=== LSTM WITH Tensorflow.NET ===");
Console.WriteLine();

clock.Start();
var ((xTrain, yTrain), (xTest, yTest)) = keras.datasets.imdb.load_data(num_words: 10000, seed: 42);
clock.Stop();
var loadTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data loaded and split in {loadTime} ms");

clock.Restart();
var layers = keras.layers;
var model = keras.Sequential(new List<Tensorflow.Keras.ILayer>
{
    layers.Embedding(10000, 128, mask_zero: true),
    layers.LSTM(128),
    layers.Dense(1, activation: "sigmoid")
});
model.compile(optimizer: "adam", loss: "binary_crossentropy", metrics: new[] { "accuracy" });
var modelFit = model.fit(xTrain, yTrain);
clock.Stop();
var trainingTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model trained in {trainingTime} ms");

clock.Restart();
var metrics = model.evaluate(xTest, yTest);
clock.Stop();
var evaluationTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model evaluated in {evaluationTime} ms");

Console.WriteLine();
Console.WriteLine("-- RESULTS --");
var totalElapsedTime = loadTime + trainingTime + evaluationTime;
Console.WriteLine($"Total elapsed time: {totalElapsedTime} ms");
Console.WriteLine($"Average distance: {metrics}");
Console.WriteLine();

Console.WriteLine("Saving model...");
//context.Model.Save(model, train.Schema, "./CustomerClusteringModel.zip");
Console.WriteLine("Done.");