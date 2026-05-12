using Microsoft.ML;
using ML.Clustering.Domain;
using System.Globalization;

var context = new MLContext(seed: 42);

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

var split = context.Data.TrainTestSplit(customers, testFraction: 0.2);
IDataView train = split.TrainSet;
IDataView test = split.TestSet;

IEstimator<ITransformer> pipeline = context.Transforms.Concatenate("Features", nameof(RfmCustomer.Recency), nameof(RfmCustomer.Frequency), nameof(RfmCustomer.Monetary))
    .Append(context.Transforms.NormalizeMinMax("Features"))
    .Append(context.Clustering.Trainers.KMeans("Features", numberOfClusters: 4));

ITransformer model = pipeline.Fit(train);

IDataView predictions = model.Transform(test);
var metrics = context.Clustering.Evaluate(predictions);
Console.WriteLine($"Average Distance: {metrics.AverageDistance}");