using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.IO;
using System.Linq;
using CreditScore.ML.Helper;
using CreditScore.ML.Trainers;
using CreditScore.ML.Trainers.Contracts;
using CreditScore_ML;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using static Microsoft.ML.DataOperationsCatalog;

namespace CreditScore.ML
{
    public class Program
    {
        private static readonly string _modelPath = @"../../../..";
        
        static void Main(string[] args)
        {

            var serviceProvider = BuildServiceProvider();
            IEnumerable<ITrainerProcess> trainers = serviceProvider.GetService<IEnumerable<ITrainerProcess>>();
            string origin = FileHelpers.GetAbsolutePath(@"Resources\3-T1_Unbalanced.csv");
            string undersampling = FileHelpers.GetAbsolutePath(@"Resources\4-T1_Undersampling.csv");
            string ovesampling = FileHelpers.GetAbsolutePath(@"Resources\5-T1_Oversampling.csv");

            string validationT2SetPath = FileHelpers.GetAbsolutePath(@"Resources\T2.csv");
            string validationT3SetPath = FileHelpers.GetAbsolutePath(@"Resources\T3.csv");

            #region "STEP 1: Common data loading configuration"
            List<CreditScoringModel> validationSetList = FileHelpers.ReadCsv(validationT2SetPath);

            //// Load the bankruptcy data as an IDataView.
            var _mlContext = new MLContext(seed: 1);
            IDataView validationData = _mlContext.Data.LoadFromEnumerable(validationSetList);

            IDataView data = _mlContext.Data.LoadFromTextFile<CreditScoringModel>(ovesampling, separatorChar: ',', hasHeader: true);
            #endregion
            foreach (ITrainerProcess trainerProcess in trainers)
            {
                trainerProcess.Process(data, validationData);
            }
        }

        private static ServiceProvider BuildServiceProvider()
        {
            MLContext mLContext = new MLContext(seed: 1);
            return new ServiceCollection()
                .AddSingleton<ITrainerProcess, FastTreeTrainer>(p =>
                {
                    // Create a new context for ML.NET operations. It can be used for
                    // exception tracking and logging, as a catalog of available operations
                    // and as the source of randomness.
                    return new FastTreeTrainer(mLContext);
                })
                .AddSingleton<ITrainerProcess, FastForestTrainer>(p =>
                {
                    // Create a new context for ML.NET operations. It can be used for
                    // exception tracking and logging, as a catalog of available operations
                    // and as the source of randomness.
                    return new FastForestTrainer(mLContext);
                })
                .AddSingleton<ITrainerProcess, SgdCalibratedTrainer>(p =>
                {
                    // Create a new context for ML.NET operations. It can be used for
                    // exception tracking and logging, as a catalog of available operations
                    // and as the source of randomness.
                    return new SgdCalibratedTrainer(mLContext);
                })
                .AddSingleton<ITrainerProcess, LinearSvmTrainer>(p =>
                {
                    // Create a new context for ML.NET operations. It can be used for
                    // exception tracking and logging, as a catalog of available operations
                    // and as the source of randomness.
                    return new LinearSvmTrainer(mLContext);
                })
                .AddSingleton<ITrainerProcess, AveragedPerceptronTrainer>(p =>
                {
                    // Create a new context for ML.NET operations. It can be used for
                    // exception tracking and logging, as a catalog of available operations
                    // and as the source of randomness.
                    return new AveragedPerceptronTrainer(mLContext);
                })
                .BuildServiceProvider();
        }

        #region "old"
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }

       

        public static class PermutationFeatureImportance
        {
            public static void Example()
            {
                // STEP 1: Common data loading configuration
                // Create a new context for ML.NET operations. It can be used for
                // exception tracking and logging, as a catalog of available operations
                // and as the source of randomness.
                var mlContext = new MLContext(seed: 1);
                //IDataView data = mlContext.Data.LoadFromTextFile<CreditScoringModel>(@"C:\Users\omar.hassani\Documents\Personal\PFM\CsvData\T1.csv", ';', true);

                List<CreditScoringModel> samples = FileHelpers.ReadCsv(@"C:\Users\omar.hassani\Documents\Personal\PFM\CsvData\T1.csv");
                // Create sample data.
                //IEnumerable<Data> samples = GenerateData();

                //// Load the sample data as an IDataView.
                IDataView data = mlContext.Data.LoadFromEnumerable(samples);

                // Define a training pipeline that concatenates features into a vector,
                // normalizes them, and then trains a linear model.
                var featureColumns = GetColumnName(data).ToArray();
                //var featureColumns = new string[] { nameof(Data.Feature1), nameof(Data.Feature2) };
                //var pipeline = FeaturizeEstimator(mlContext)
                //    .Append(mlContext.BinaryClassification.Trainers
                //    .SdcaLogisticRegression(labelColumnName: "Label"));

                var pipeline = mlContext.Transforms
                                .Concatenate("Features", featureColumns)
                                .Append(mlContext.Transforms.NormalizeMinMax("Features"))
                                .Append(mlContext.BinaryClassification.Trainers
                                .SdcaLogisticRegression(labelColumnName: "Label"));

                // Fit the pipeline to the data.
                var model = pipeline.Fit(data);

                // Transform the dataset.
                var transformedData = model.Transform(data);

                // Extract the predictor.
                var linearPredictor = model.LastTransformer;

                var weight = linearPredictor.Model.SubModel.Weights;
                Console.WriteLine("Feature#Model Weight");
                weight.OrderByDescending(c => c);
                for (int i = 0; i < weight.Count; i++)
                {
                    Console.WriteLine("{0}#{1:0.00}",
                        "V" + i + 1,
                        linearPredictor.Model.SubModel.Weights[i]);
                }

                // Compute the permutation metrics for the linear model using the
                // normalized data.
                var permutationMetrics = mlContext.BinaryClassification
                    .PermutationFeatureImportance(linearPredictor, transformedData,
                    permutationCount: 30, labelColumnName: "Label");

                // Now let's look at which features are most important to the model
                // overall. Get the feature indices sorted by their impact on AUC.
                var sortedIndices = permutationMetrics
                    .Select((metrics, index) => new { index, metrics.AreaUnderRocCurve })
                    .OrderByDescending(
                    feature => Math.Abs(feature.AreaUnderRocCurve.Mean))
                    .Select(feature => feature.index);

                Console.WriteLine("Feature#Model Weight#Change in AUC#Change in ACC"
                    + "#95% Confidence in the Mean Change in AUC");
                var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray();
                var acc = permutationMetrics.Select(x => x.Accuracy).ToArray();

                foreach (int i in sortedIndices)
                {
                    Console.WriteLine("{0}#{1:0.00}#{2:G4}#{3:G4}#{4:G4}",
                        featureColumns[i],
                        linearPredictor.Model.SubModel.Weights[i],
                        auc[i].Mean,
                        acc[i].Mean,
                        1.96 * auc[i].StandardError);
                }

                // Expected output:
                //  Feature     Model Weight Change in AUC  95% Confidence in the Mean Change in AUC
                //  Feature2        35.15     -0.387        0.002015
                //  Feature1        17.94     -0.1514       0.0008963
            }

            public static EstimatorChain<NormalizingTransformer> FeaturizeEstimator(MLContext mlContext)
            {
                EstimatorChain<NormalizingTransformer> dataProcessPipeline =
                    mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(CreditScoringModel.CNAE))
                                              .Append(mlContext.Transforms.NormalizeMinMax("Features"));
                return dataProcessPipeline;
            }
            private static List<string> GetColumnName(IDataView data)
            {
                List<string> featureColumns = new List<string>();
                foreach (DataViewSchema.Column item in data.Schema.Where(c => c.Name != "Label" && c.Name != "Label" && c.Name != "CNAE"))
                {
                    featureColumns.Add(item.Name);
                }
                return featureColumns;
            }

            private class Data
            {
                public bool Label { get; set; }

                public float Feature1 { get; set; }

                public float Feature2 { get; set; }
            }

            /// <summary>
            /// Generate an enumerable of Data objects, creating the label as a simple
            /// linear combination of the features.
            /// </summary>
            /// <param name="nExamples">The number of examples.</param>
            /// <param name="bias">The bias, or offset, in the calculation of the label.
            /// </param>
            /// <param name="weight1">The weight to multiply the first feature with to
            /// compute the label.</param>
            /// <param name="weight2">The weight to multiply the second feature with to
            /// compute the label.</param>
            /// <param name="seed">The seed for generating feature values and label
            /// noise.</param>
            /// <returns>An enumerable of Data objects.</returns>
            private static IEnumerable<Data> GenerateData(int nExamples = 10000,
                double bias = 0, double weight1 = 1, double weight2 = 2, int seed = 1)
            {
                var rng = new Random(seed);
                for (int i = 0; i < nExamples; i++)
                {
                    var data = new Data
                    {
                        Feature1 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                        Feature2 = (float)(rng.Next(10) * (rng.NextDouble() - 0.5)),
                    };

                    // Create a noisy label.
                    var value = (float)(bias + weight1 * data.Feature1 + weight2 *
                        data.Feature2 + rng.NextDouble() - 0.5);

                    data.Label = Sigmoid(value) > 0.5;
                    yield return data;
                }
            }

            private static double Sigmoid(double x) => 1.0 / (1.0 + Math.Exp(-1 * x));
        }
        #endregion


    }
}
