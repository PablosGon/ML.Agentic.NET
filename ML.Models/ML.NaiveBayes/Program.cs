using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML;
using ML.NaiveBayes.Domain;
using System.Diagnostics;
using System.Globalization;

var clock = new Stopwatch();

Console.WriteLine("=== NAIVE BAYES WITH ML.NET ===");
Console.WriteLine();
var context = new MLContext(seed: 42);

clock.Start();
using var reader = new StreamReader("./Phishing_Email.csv");
var config = new CsvConfiguration(CultureInfo.InvariantCulture)
{
    HasHeaderRecord = true,
    HeaderValidated = null,
};
using var csvReader = new CsvReader(reader, config);
var emailList = csvReader.GetRecords<Email>().ToList();
IDataView tickets = context.Data.LoadFromEnumerable(emailList);
clock.Stop();
var loadTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data loaded in {loadTime} ms");

clock.Restart();
var split = context.Data.TrainTestSplit(tickets, testFraction: 0.2);
IDataView train = split.TrainSet;
IDataView test = split.TestSet;
clock.Stop();
var splitTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Data split into training and test sets in {splitTime} ms");

clock.Restart();
IEstimator<ITransformer> pipeline = context.Transforms.Conversion.MapValueToKey($"{nameof(Email.Type)}Encoded", nameof(Email.Type))
    .Append(context.Transforms.Text.FeaturizeText($"{nameof(Email.Text)}Featurized", options: new Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Options()
    {
        CharFeatureExtractor = null,
        CaseMode = Microsoft.ML.Transforms.Text.TextNormalizingEstimator.CaseMode.Lower,
        StopWordsRemoverOptions = new Microsoft.ML.Transforms.Text.StopWordsRemovingEstimator.Options()
        {
            Language = Microsoft.ML.Transforms.Text.TextFeaturizingEstimator.Language.English
        },
        WordFeatureExtractor = new Microsoft.ML.Transforms.Text.WordBagEstimator.Options()
        {
            NgramLength = 2,
            MaximumNgramsCount = [5000, 5000],
        }
    }, [nameof(Email.Text)]))
    .Append(context.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: $"{nameof(Email.Type)}Encoded", featureColumnName: $"{nameof(Email.Text)}Featurized"));

ITransformer model = pipeline.Fit(train);
clock.Stop();
var trainingTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model trained in {trainingTime} ms");

clock.Restart();
IDataView predictions = model.Transform(test);
var metrics = context.MulticlassClassification.Evaluate(predictions, $"{nameof(Email.Type)}Encoded");
clock.Stop();
var evaluationTime = clock.ElapsedMilliseconds;
Console.WriteLine($"Model evaluated in {evaluationTime} ms");

Console.WriteLine();
Console.WriteLine("-- RESULTS --");
var totalElapsedTime = loadTime + splitTime + trainingTime + evaluationTime;
Console.WriteLine($"Total elapsed time: {totalElapsedTime} ms");
Console.WriteLine($"Accuracy: {metrics.MacroAccuracy}");
Console.WriteLine($"{metrics.ConfusionMatrix.GetFormattedConfusionTable()}");
Console.WriteLine();

Console.WriteLine("Saving model...");
context.Model.Save(model, train.Schema, "./PhishingDetectionModel.zip");
Console.WriteLine("Done.");