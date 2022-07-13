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
    public class SdcaLogisticRegressionTrainer : ITrainerProcess
    {
        private static readonly string _modelPath = @"../../../..";
        private readonly MLContext _mlContext;

        public SdcaLogisticRegressionTrainer(MLContext mlContext)
        {
            _mlContext = mlContext;
        }
        public void Process(IDataView trainingDataView, IDataView validationDataView = null)
        {

            #region "STEP 2:  Definition of binary clasification trainers/algorithms to use"

            // STEP 2:  Definition of binary clasification trainers/algorithms to use    
            TrainersAgregator trainersAgregator = new TrainersAgregator(_mlContext, _modelPath);
            (string name, IEstimator<ITransformer> value) trainingModel = ("SdcaLogisticRegression", _mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Label", featureColumnName: "Features"));

            EstimatorChain<NormalizingTransformer> pipelineBase = trainersAgregator.FeaturizeEstimator();
            #endregion

            #region "STEP 3.1:  Generate Trainers Model"

            // STEP 3.1:  Generate Trainers Model
            //trainersAgregator.GenerateTrainersModel(trainingDataView, validationDataView, trainingModel, pipelineBase);

            #endregion

            #region "STEP 3.2:  Generate Trainers Model"

            // STEP 3.2:  Generate Trainers Model
            trainersAgregator.GenerateTrainersModelCrossValidation(trainingDataView, trainingModel, pipelineBase);

            #endregion
            #region "STEP 4:  Evaluate the model and show accuracy stats"

            // STEP 3:  Generate Trainers Model
            //trainersAgregator.EvaluateModel(trainingModel, validationSetPath);

            #endregion
            #region "STEP 4:  Featurise attribut"

            var featureColumns = FeatureHelper.GetColumnName(trainingDataView).ToArray();

            var pipeline = _mlContext.Transforms
                            .Concatenate("Features", featureColumns)
                            .Append(_mlContext.Transforms.NormalizeMinMax("Features"))
                            .Append(_mlContext.BinaryClassification.Trainers
                            .SdcaLogisticRegression(labelColumnName: "Label"));

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

            #endregion
        }
    }
}
