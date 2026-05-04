using Microsoft.ML;
using ML.Regression.Domain;
using System.Diagnostics;

var clock = new Stopwatch();

Console.WriteLine("=== LINEAR REGRESSION WITH ML.NET ===");
Console.WriteLine();
MLContext context = new MLContext(seed: 42);

clock.Start();
IDataView housePricing = context.Data.LoadFromTextFile<HousePricing>("./HousePrice.csv", separatorChar: ',', hasHeader: true);
clock.Stop();
var loadTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data loaded in {loadTime} ms");

clock.Restart();
var split = context.Data.TrainTestSplit(housePricing, testFraction: 0.2);
IDataView trainingData = split.TrainSet;
IDataView testData = split.TestSet;
clock.Stop();
var splitTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data split into training and test sets in {splitTime} ms");

clock.Restart();
IEstimator<ITransformer> pipeline = context.Transforms.Categorical.OneHotEncoding(
    $"{nameof(HousePricing.NeighbourhoodQuality)}Encoded", nameof(HousePricing.NeighbourhoodQuality))
    .Append(context.Transforms.Concatenate(
        "Features",
        nameof(HousePricing.SquareFootage),
        nameof(HousePricing.NumBedrooms),
        nameof(HousePricing.NumBathrooms),
        nameof(HousePricing.LotSize),
        nameof(HousePricing.GarageSize),
        $"{nameof(HousePricing.NeighbourhoodQuality)}Encoded"))
    .Append(context.Transforms.NormalizeMeanVariance("Features"))
    .Append(context.Regression.Trainers.Sdca(nameof(HousePricing.HousePrice), "Features"));
ITransformer model = pipeline.Fit(trainingData);
clock.Stop();
var trainingTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model trained in {trainingTime} ms");

clock.Restart();
IDataView predictions = model.Transform(testData);
var metrics = context.Regression.Evaluate(predictions, nameof(HousePricing.HousePrice));
clock.Stop();
var evaluationTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model evaluated in {evaluationTime} ms");

Console.WriteLine();
Console.WriteLine("-- RESULTS --");
var totalElapsedTime = loadTime + splitTime + trainingTime + evaluationTime;
Console.WriteLine($"Total elapsed time: {totalElapsedTime} ms");
Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
Console.WriteLine($"MAE: {metrics.MeanAbsoluteError}");
Console.WriteLine($"R^2: {metrics.RSquared}");
Console.WriteLine();

Console.WriteLine("Saving model...");
context.Model.Save(model, trainingData.Schema, "./HousePricingModel.zip");
Console.WriteLine("Done.");