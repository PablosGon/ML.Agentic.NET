using Microsoft.ML;
using ML.Regression.Domain;

MLContext context = new MLContext();

IDataView housePricing = context.Data.LoadFromTextFile<HousePricing>("./HousePrice.csv", separatorChar: ',', hasHeader: true);

var split = context.Data.TrainTestSplit(housePricing, testFraction: 0.2);
IDataView trainingData = split.TrainSet;
IDataView testData = split.TestSet;

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

IDataView predictions = model.Transform(testData);
var metrics = context.Regression.Evaluate(predictions, nameof(HousePricing.HousePrice));

Console.WriteLine($"RMSE: {metrics.RootMeanSquaredError}");
Console.WriteLine($"MAE: {metrics.MeanAbsoluteError}");
Console.WriteLine($"R^2: {metrics.RSquared}");