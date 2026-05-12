using CsvHelper;
using CsvHelper.Configuration;
using Microsoft.ML;
using ML.NaiveBayes.Domain;
using System.Globalization;

var context = new MLContext(seed: 42);

using var reader = new StreamReader("./Phishing_Email.csv");
var config = new CsvConfiguration(CultureInfo.InvariantCulture)
{
    HasHeaderRecord = true,
    HeaderValidated = null,
};
using var csvReader = new CsvReader(reader, config);
var ticketsFromFile = csvReader.GetRecords<Email>();

// Materialize CSVHelper results (CsvHelper correctly preserves newlines inside quoted fields)
var emailList = ticketsFromFile.ToList();
// Create an IDataView from the parsed records instead of re-reading the file with LoadFromTextFile
IDataView tickets = context.Data.LoadFromEnumerable(emailList);

var split = context.Data.TrainTestSplit(tickets, testFraction: 0.2);
IDataView train = split.TrainSet;
IDataView test = split.TestSet;

Console.WriteLine($"Data split into training and test sets in {"X"} ms");

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

Console.WriteLine($"Model trained in {"X"} ms");

IDataView predictions = model.Transform(test);
var metrics = context.MulticlassClassification.Evaluate(predictions, $"{nameof(Email.Type)}Encoded");

Console.WriteLine(metrics.MacroAccuracy);