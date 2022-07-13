using CreditScore.ML.Helper;
using CreditScore.ML.Trainers.Contracts;
using CreditScore.ML.Trainers.Helpers;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Transforms;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CreditScore.ML.Trainers
{
    public class NaiveBayes_Trainer : ITrainerProcess
    {
        private static readonly string _modelPath = @"../../../..";
        private readonly MLContext _mlContext;
        public NaiveBayes_Trainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }
        public void Process(string trainingSetPath, string validationSetPath)
        {
            // Create a new context for ML.NET operations. It can be used for
            // exception tracking and logging, as a catalog of available operations
            // and as the source of randomness.

            #region "STEP 1:  Common data loading configuration "

            // STEP 1: Common data loading configuration
            List<CreditScoringModel> samples = FileHelpers.ReadCsv(trainingSetPath);

            //// Load the bankruptcy data as an IDataView.
            IDataView dataView = _mlContext.Data.LoadFromEnumerable(samples);
            #endregion

            #region "STEP 2:  Definition of binary clasification trainers/algorithms to use"

            // STEP 2:  Definition of binary clasification trainers/algorithms to use    
            TrainersAgregator trainersAgregator = new TrainersAgregator(_mlContext, _modelPath);
            (string name, IEstimator<ITransformer> value) trainingModel = ("NaiveBayes", _mlContext.MulticlassClassification.Trainers.NaiveBayes(labelColumnName: "Bankruptcy", featureColumnName: "Features"));

            EstimatorChain<NormalizingTransformer> pipelineBase = trainersAgregator.FeaturizeEstimator();

            #endregion

            #region "STEP 3.1:  Generate Trainers Model"

            // STEP 3.1:  Generate Trainers Model
            trainersAgregator.GenerateTrainersModel(dataView, trainingModel, pipelineBase, false);

            #endregion

            #region "STEP 3.2:  Generate Trainers Model"

            // STEP 3.2:  Generate Trainers Model
            //trainersAgregator.GenerateTrainersModelCrossValidation(dataView, trainingModel);

            #endregion
            #region "STEP 4:  Evaluate the model and show accuracy stats"

            // STEP 3:  Generate Trainers Model
            //trainersAgregator.EvaluateModel(trainingModel, validationSetPath);

            #endregion
            #region "STEP 4:  Featurise attribut"

            var featureColumns = FeatureHelper.GetColumnName(dataView).ToArray();

            var pipeline = _mlContext.Transforms
                            .Concatenate("Features", featureColumns)
                            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                            .Append(_mlContext.MulticlassClassification.Trainers
                            .NaiveBayes(labelColumnName: "Bankruptcy"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(dataView);

            // Transform the dataset.
            var transformedData = model.Transform(dataView);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.MulticlassClassification
                .PermutationFeatureImportance(linearPredictor, transformedData,
                permutationCount: 30, labelColumnName: "Bankruptcy");

            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on MicroAccuracy mean.

            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.MicroAccuracy })
                .OrderByDescending(feature => Math.Abs(feature.MicroAccuracy.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature\tChange in MicroAccuracy\tLogLoss\t95% Confidence in "
                + "the Mean Change in MicroAccuracy");
            var microAccuracies = permutationMetrics.Select(x => x.MicroAccuracy).ToArray();
            var LogLoss = permutationMetrics.Select(x => x.LogLoss).ToArray();

            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}\t{1:G4}\t{2:G4}",
                    featureColumns[i],
                    microAccuracies[i].Mean,
                    LogLoss[i].Mean,
                    1.96 * microAccuracies[i].StandardError);
            }

            #endregion
        }
    }
}