using CreditScore.ML.Model;
using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CreditScore.ML.Agregators
{
    public class ModelScoringTester
    {
        public MLContext _mlContext { get; }

        public ModelScoringTester(MLContext mlContext)
        {
            _mlContext = mlContext ?? new MLContext(seed: 1);

        }
        public static void VisualizeSomePredictions(MLContext _mlContext,
                                                    string modelName,
                                                    string testDataLocation,
                                                    PredictionEngine<BankruptcyObservation, BankruptcyPrediction> predEngine,
                                                    int numberOfPredictions)
        {
            //Make a few prediction tests 
            // Make the provided number of predictions and compare with observed data from the test dataset
            List<BankruptcyObservation> testData = ReadSampleDataFromCsvFile(testDataLocation, numberOfPredictions, true);

            for (int i = 0; i < numberOfPredictions; i++)
            {
                //Score
                var resultprediction = predEngine.Predict(testData[i]);

                ConsoleHelper.ConsoleWriterSection(resultprediction.Bankruptcy.ToString());
            }

        }

        //This method is using regular .NET System.IO.File and LinQ to read just some sample data to test/predict with 
        public static List<BankruptcyObservation> ReadSampleDataFromCsvFile(string dataLocation, int numberOfRecordsToRead, bool withHeather = true)
        {
            var allData = File.ReadLines(dataLocation)
                .Skip(withHeather ? 1 : 0)
                .Where(x => !string.IsNullOrWhiteSpace(x))
                .Select(x => x.Split(';'))
                .Select(x => new BankruptcyObservation()
                {
                    V1 = float.Parse(x[0], CultureInfo.InvariantCulture),
                    V2 = float.Parse(x[1], CultureInfo.InvariantCulture),
                    V3 = float.Parse(x[2], CultureInfo.InvariantCulture),
                    V4 = float.Parse(x[3], CultureInfo.InvariantCulture),
                    V5 = float.Parse(x[4], CultureInfo.InvariantCulture),
                    V6 = float.Parse(x[5], CultureInfo.InvariantCulture),
                    V7 = float.Parse(x[6], CultureInfo.InvariantCulture),
                    V8 = float.Parse(x[7], CultureInfo.InvariantCulture),
                    V9 = float.Parse(x[8], CultureInfo.InvariantCulture),
                    V10 = float.Parse(x[9], CultureInfo.InvariantCulture),
                    V11 = float.Parse(x[10], CultureInfo.InvariantCulture),
                    V12 = float.Parse(x[11], CultureInfo.InvariantCulture),
                    V13 = float.Parse(x[12], CultureInfo.InvariantCulture),
                    CNAE = x[13].ToString()
                });
            
            return allData
                .Take(numberOfRecordsToRead > 0 ? numberOfRecordsToRead : allData.Count())
                .ToList();
        }
    }
}
