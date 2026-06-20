using System.Diagnostics;
using static Tensorflow.KerasApi;

var clock = new Stopwatch();

Console.WriteLine("=== LSTM WITH Tensorflow.NET ===");
Console.WriteLine();

clock.Start();
var ((xTrain, yTrain), (xTest, yTest)) = keras.datasets.imdb.load_data(num_words: 10000, seed: 42);
clock.Stop();
var loadTime = clock.ElapsedMilliseconds;
var xTrainList = xTrain.Select(x => x.ToArray<int>()).ToList();
var xTestList = xTest.Select(x => x.ToArray<int>()).ToList();

Console.WriteLine($"Data loaded and split in {loadTime} ms");

clock.Restart();
var maxLength = 256;
xTrain = keras.preprocessing.sequence.pad_sequences(xTrainList, maxlen: maxLength);
xTest = keras.preprocessing.sequence.pad_sequences(xTestList, maxlen: maxLength);
clock.Stop();
var paddingTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data padded in {paddingTime} ms");


clock.Restart();
var layers = keras.layers;
var model = keras.Sequential(new List<Tensorflow.Keras.ILayer>
{
    layers.Embedding(10000, 128, mask_zero: true),
    layers.LSTM(128),
    layers.Dense(1, activation: "sigmoid")
});
model.compile(optimizer: "adam", loss: "binary_crossentropy", metrics: new[] { "accuracy" });
var modelFit = model.fit(xTrain, yTrain, epochs: 3, batch_size: 64, validation_split: 0.2f);
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
Console.WriteLine($"Accuracy score : {metrics["accuracy"]}");
Console.WriteLine();

Console.WriteLine("Saving model...");
//context.Model.Save(model, train.Schema, "./CustomerClusteringModel.zip");
Console.WriteLine("Done.");