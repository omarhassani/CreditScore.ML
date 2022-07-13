using CreditScore.ML.Helper;
using CreditScore.ML.Trainers.Contracts;
using CreditScore.ML.Trainers.Helpers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using static Microsoft.ML.DataOperationsCatalog;

namespace CreditScore.ML
{
    public class NaiveBayesTrainer : ITrainerProcess
    {
        private readonly string _modelPath = @"../../../..";
        private readonly MLContext _mlContext;
        private string trainer = "NaiveBayesTrainer";
        public NaiveBayesTrainer(MLContext mlContext)
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness.
            _mlContext = mlContext;
        }

        public IReadOnlyList<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> CrossValidation(IDataView dataView, IEstimator<ITransformer> pipeline)
        {
            IReadOnlyList<TrainCatalogBase.CrossValidationResult<MulticlassClassificationMetrics>> crossValidationResults = _mlContext.MulticlassClassification.CrossValidate(data: dataView, estimator: pipeline, numberOfFolds: 10, labelColumnName: "Label");

            ConsoleHelper.PrintMulticlassClassificationFoldsAverageMetrics(trainer, crossValidationResults);

            return crossValidationResults;

        }

        /// <summary>
        /// <summary>
        /// Build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="_mlContext"></param>
        /// <returns></returns>
        public IEstimator<ITransformer> BuildPipeline()
        {
            // Data process configuration with pipeline data transformations
            var pipeline = _mlContext.Transforms.ReplaceMissingValues(new[] { new InputOutputColumnPair(@"V1", @"V1"), new InputOutputColumnPair(@"V2", @"V2"), new InputOutputColumnPair(@"V3", @"V3"), new InputOutputColumnPair(@"V4", @"V4"), new InputOutputColumnPair(@"V5", @"V5"), new InputOutputColumnPair(@"V6", @"V6"), new InputOutputColumnPair(@"V7", @"V7"), new InputOutputColumnPair(@"V8", @"V8"), new InputOutputColumnPair(@"V9", @"V9"), new InputOutputColumnPair(@"V10", @"V10"), new InputOutputColumnPair(@"V11", @"V11"), new InputOutputColumnPair(@"V12", @"V12"), new InputOutputColumnPair(@"V13", @"V13") })
                                    .Append(_mlContext.Transforms.Text.FeaturizeText(@"CNAE", @"CNAE"))
                                    .Append(_mlContext.Transforms.Concatenate(@"Features", new[] { @"V1", @"V2", @"V3", @"V4", @"V5", @"V6", @"V7", @"V8", @"V9", @"V10", @"V11", @"V12", @"V13", @"CNAE" }))
                                    .Append(_mlContext.Transforms.Conversion.MapValueToKey(@"Label", @"Label"))
                                    .Append(_mlContext.Transforms.NormalizeMinMax(@"Features", @"Features"))
                                    .Append(_mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: @"Label", featureColumnName: @"Features"))
                                    .Append(_mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel", @"PredictedLabel"));

            return pipeline;
        }

        public void Process(IDataView trainingDataView, IDataView validationDataView = null)
        {
            #region "STEP 2:  Definition of binary clasification trainers/algorithms to use"

            // Load the sample data as an IDataView.

            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var featureColumns = FileHelpers.GetColumnNameExcept_Label_Cnae(trainingDataView);


            var pipeline = _mlContext.Transforms
                .Concatenate("Features", featureColumns.ToArray())
                .Append(_mlContext.Transforms.Conversion.MapValueToKey("Label"))
                .Append(_mlContext.Transforms.Text.FeaturizeText(@"CNAE", @"CNAE"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                .Append(_mlContext.MulticlassClassification.Trainers
                .NaiveBayes(featureColumnName: @"Features"));

            var trainedModel = pipeline.Fit(trainingDataView);

            ConsoleHelper.WriteLog($"===== Evaluating {trainer} Model's accuracy with Test data =====");
            IDataView predictions = trainedModel.Transform(validationDataView);


            MulticlassClassificationMetrics metrics = _mlContext.MulticlassClassification.Evaluate(data: predictions, labelColumnName: "Label");
            ConsoleHelper.PrintMulticlassClassificationMetrics(trainer, metrics);

            // Extract the predictor.
            var linearPredictor = trainedModel.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = this._mlContext.MulticlassClassification
                    .PermutationFeatureImportance(linearPredictor, predictions,
                    permutationCount: 30, labelColumnName: "Label");


            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on
            // microaccuracy.
            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.MicroAccuracy })
                .OrderByDescending(feature => Math.Abs(feature.MicroAccuracy.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#" +
                "Change in MicroAccuracy#" +
                "Change in MacroAccuracy#" +
                "Change in TopKAccuracy#" +
                "Change in LogLoss#" +
                "Change in LogLossReduction#" +
                "96% Confidence in the Mean Change in MicroAccuracy");

            var microAccuracy = permutationMetrics.Select(x => x.MicroAccuracy).ToArray();
            var logLossReduction = permutationMetrics.Select(x => x.LogLossReduction).ToArray();
            var logLoss = permutationMetrics.Select(x => x.LogLoss).ToArray();
            var macroAccuracy = permutationMetrics.Select(x => x.MacroAccuracy).ToArray();
            var topKAccuracy = permutationMetrics.Select(x => x.TopKAccuracy).ToArray();

            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1}#{2}#{3}#{4}#{5}#{6}",
                    featureColumns[i],
                    microAccuracy[i].Mean,
                    macroAccuracy[i].Mean,
                    topKAccuracy[i].Mean,
                    logLoss[i].Mean,
                    logLossReduction[i].Mean,
                    1.96 * microAccuracy[i].StandardError);
            }

            #endregion
        }
    }
}