using Microsoft.ML;
using ML.Clustering.Domain;
using System.Diagnostics;
using System.Globalization;

var clock = new Stopwatch();

Console.WriteLine("=== K-MEANS WITH ML.NET ===");
Console.WriteLine();
var context = new MLContext(seed: 42);

clock.Start();
var reader = new StreamReader("./online_retail_II.csv");
var csvConfig = new CsvHelper.Configuration.CsvConfiguration(new CultureInfo("es-ES"))
{
    Delimiter = ";",
    HasHeaderRecord = true,
    HeaderValidated = null,
};
var csvReader = new CsvHelper.CsvReader(reader, csvConfig);
var transactions = csvReader.GetRecords<Transaction>().ToList();
var latestDate = transactions.Max(x => x.InvoiceDate);
var rfmCustomers = transactions
    .GroupBy(x => x.CustomerId)
    .Select(x => new RfmCustomer
    {
        Recency = (float)(latestDate - x.Max(r => r.InvoiceDate)).TotalDays,
        Frequency = x.Select(r => r.Invoice).Distinct().Count(),
        Monetary = (float) x.Sum(r => r.Price * r.Quantity)
    });

IDataView customers = context.Data.LoadFromEnumerable(rfmCustomers);
clock.Stop();
var loadTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data loaded and formatted in {loadTime} ms");

clock.Restart();
var split = context.Data.TrainTestSplit(customers, testFraction: 0.2);
IDataView train = split.TrainSet;
IDataView test = split.TestSet;
clock.Stop();
var splitTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data split into training and test sets in {splitTime} ms");

clock.Restart();
IEstimator<ITransformer> pipeline = context.Transforms.Concatenate("Features", nameof(RfmCustomer.Recency), nameof(RfmCustomer.Frequency), nameof(RfmCustomer.Monetary))
    .Append(context.Transforms.NormalizeMinMax("Features"))
    .Append(context.Clustering.Trainers.KMeans("Features", numberOfClusters: 4));

ITransformer model = pipeline.Fit(train);
clock.Stop();
var trainingTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model trained in {trainingTime} ms");

clock.Restart();
IDataView predictions = model.Transform(test);
var metrics = context.Clustering.Evaluate(predictions);
Console.WriteLine($"Average Distance: {metrics.AverageDistance}");
clock.Stop();
var evaluationTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model evaluated in {evaluationTime} ms");

Console.WriteLine();
Console.WriteLine("-- RESULTS --");
var totalElapsedTime = loadTime + splitTime + trainingTime + evaluationTime;
Console.WriteLine($"Total elapsed time: {totalElapsedTime} ms");
Console.WriteLine($"Average distance: {metrics.AverageDistance}");
Console.WriteLine();

Console.WriteLine("Saving model...");
context.Model.Save(model, train.Schema, "./CustomerClusteringModel.zip");
Console.WriteLine("Done.");