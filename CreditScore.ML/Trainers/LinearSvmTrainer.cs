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
    public class LinearSvmTrainer : ITrainerProcess
    {
        private static readonly string _modelPath = @"../../../..";
        private readonly MLContext _mlContext;
        public LinearSvmTrainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }
        public void Process(IDataView trainingDataView, IDataView validationDataView = null)
        {
            #region "STEP 2:  Definition of binary clasification trainers/algorithms to use"

            // STEP 2:  Definition of binary clasification trainers/algorithms to use    
            TrainersAgregator trainersAgregator = new TrainersAgregator(_mlContext, _modelPath);
            (string name, IEstimator<ITransformer> value) trainingModel = ("LinearSvm", _mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label", featureColumnName: "Features"));

            var featureColumns = FeatureHelper.GetColumnName(trainingDataView).ToArray();

            EstimatorChain<NormalizingTransformer> pipelineBase = _mlContext.Transforms
                            .Concatenate("Features", featureColumns.ToArray())
                            .Append(_mlContext.Transforms.NormalizeMinMax("Features"));

            #endregion

            #region "STEP 3.1:  Generate Trainers Model"

            // STEP 3.1:  Generate Trainers Model
            //trainersAgregator.GenerateTrainersModel(trainingDataView, validationDataView, trainingModel, pipelineBase, false);

            #endregion

            #region "STEP 3.2:  Generate Trainers Model"

            // STEP 3.2:  Generate Trainers Model
            //trainersAgregator.GenerateTrainersModelCrossValidation(trainingDataView, trainingModel, pipelineBase);

            #endregion
            #region "STEP 4:  Evaluate the model and show accuracy stats"

            // STEP 3:  Generate Trainers Model
            //trainersAgregator.EvaluateModel(trainingModel, validationSetPath);

            #endregion
            #region "STEP 4:  Featurise attribut"


            Featurize(trainingDataView, trainingModel, featureColumns, pipelineBase);

            #endregion
        }

        private void Featurize(IDataView trainingDataView, (string name, IEstimator<ITransformer> value) trainingModel, string[] featureColumns, EstimatorChain<NormalizingTransformer> pipelineBase)
        {
            var pipeline = pipelineBase
                                        .Append(_mlContext.BinaryClassification.Trainers.LinearSvm(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(trainingDataView);

            // Transform the dataset.
            var transformedData = model.Transform(trainingDataView);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.BinaryClassification
                .PermutationFeatureImportance(linearPredictor, transformedData,
                permutationCount: 30, labelColumnName: "Label");

            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on AUC.
            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.Accuracy })
                .OrderByDescending(
                feature => Math.Abs(feature.Accuracy.Mean))
                .Select(feature => feature.index);
            Console.WriteLine(trainingModel.name);

            Console.WriteLine("#Feature#" +
                "Model Weight#" +
                "Change in AUC#" +
                "Change in ACC#" +
                "Change in areaUnderPrecisionRecallCurve#" +
                "Change in f1Score#" +
                "Change in negativePrecision#" +
                "Change in positivPrecision#" +
                "Change in positiveRecall#" +
                "Change in negativeRecall#" +
                "95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray();
            var acc = permutationMetrics.Select(x => x.Accuracy).ToArray();
            var areaUnderPrecisionRecallCurve = permutationMetrics.Select(x => x.AreaUnderPrecisionRecallCurve).ToArray();
            var f1Score = permutationMetrics.Select(x => x.F1Score).ToArray();
            var negativePrecision = permutationMetrics.Select(x => x.NegativePrecision).ToArray();
            var positivPrecision = permutationMetrics.Select(x => x.PositivePrecision).ToArray();

            var positiveRecall = permutationMetrics.Select(x => x.PositiveRecall).ToArray();
            var negativeRecall = permutationMetrics.Select(x => x.NegativeRecall).ToArray();

            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:0.00}#{2:G4}#{3:G4}#{4:G4}#{5:G4}#{6:G4}#{7:G4}#{8:G4}#{9:G4}#{10:G4}",
                    featureColumns[i],
                    Math.Abs(linearPredictor.Model.Weights[i]),
                    Math.Abs(auc[i].Mean),
                    Math.Abs(acc[i].Mean),
                    Math.Abs(areaUnderPrecisionRecallCurve[i].Mean),
                    Math.Abs(f1Score[i].Mean),
                    Math.Abs(negativePrecision[i].Mean),
                    Math.Abs(positivPrecision[i].Mean),
                    Math.Abs(positiveRecall[i].Mean),
                    Math.Abs(negativeRecall[i].Mean),
                    Math.Abs(1.96 * auc[i].StandardError));
            }
        }
    }
}