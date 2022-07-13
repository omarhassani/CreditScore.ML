using CreditScore.ML.Agregators;
using CreditScore.ML.Extensions;
using CreditScore.ML.Helper;
using CreditScore.ML.Model;
using Microsoft.ML;
using Microsoft.ML.Calibrators;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Transforms;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.Collections.Immutable;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.DataOperationsCatalog;

namespace CreditScore.ML
{
    public class TrainersAgregator
    {
        private readonly MLContext _mlContext = new MLContext(seed: 1);
        private readonly string _locationPathForPersistence = null;
        private readonly string _baseDatasetsRelativePath;
        private readonly string _baseModelsRelativePath;
        public TrainersAgregator(MLContext mlContext, string locationPathForPersistence)
        {
            _mlContext = mlContext ?? new MLContext(seed: 1);
            _locationPathForPersistence = locationPathForPersistence;
            _baseDatasetsRelativePath = $@"{_locationPathForPersistence}/Data";
            _baseModelsRelativePath = $@"{_locationPathForPersistence}/MLModels";
        }

        /// <summary>
        /// Featurize Estimator with transformation, normalization, missing value
        /// </summary>
        /// <returns></returns>
        public EstimatorChain<NormalizingTransformer> FeaturizeEstimator()
        {
            EstimatorChain<NormalizingTransformer> pipeline = _mlContext.Transforms
                            .ReplaceMissingValues(new[] { new InputOutputColumnPair(@"V1", @"V1"), new InputOutputColumnPair(@"V2", @"V2"), new InputOutputColumnPair(@"V3", @"V3"), new InputOutputColumnPair(@"V4", @"V4"), new InputOutputColumnPair(@"V5", @"V5"), new InputOutputColumnPair(@"V6", @"V6"), new InputOutputColumnPair(@"V7", @"V7"), new InputOutputColumnPair(@"V8", @"V8"), new InputOutputColumnPair(@"V9", @"V9"), new InputOutputColumnPair(@"V10", @"V10"), new InputOutputColumnPair(@"V11", @"V11"), new InputOutputColumnPair(@"V12", @"V12"), new InputOutputColumnPair(@"V13", @"V13") }, MissingValueReplacingEstimator.ReplacementMode.Mean)
                           .Append(_mlContext.Transforms.Text.FeaturizeText(@"CNAE", @"CNAE"))
                           .Append(_mlContext.Transforms.Concatenate(@"Features", new[] { @"V1", @"V2", @"V3", @"V4", @"V5", @"V6", @"V7", @"V8", @"V9", @"V10", @"V11", @"V12", @"V13", @"CNAE" }))
                           .Append(_mlContext.Transforms.NormalizeMinMax(@"Features", @"Features"));

            return pipeline;
        }
        
        /// <summary>
        /// Generate Trainers Model
        /// </summary>
        /// <param name="dataView"></param>
        /// <param name="binaryClassificationLearners"></param>
        public void GenerateTrainersModel(IDataView dataView,
            (string name, IEstimator<ITransformer> value) trainer, 
            EstimatorChain<NormalizingTransformer> pipelineBase, 
            bool calibrated = false)
        {
            //Train and test generators
            TrainTestData trainTestSplit = _mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            IDataView trainingDataView = trainTestSplit.TrainSet;
            IDataView testDataView = trainTestSplit.TestSet;


            var pipeline = pipelineBase
                            .Append(trainer.value);

            // 3. Phase for Training, Evaluation and model file persistence
            // Per each regression trainer: Train, Evaluate, and Save a different model

            ConsoleHelper.WriteLog($"=============== Training the {trainer} model ===============");
                var trainedModel = pipeline.Fit(trainingDataView);

                ConsoleHelper.WriteLog($"===== Evaluating {trainer} Model's accuracy with Test data =====");
                IDataView predictions = trainedModel.Transform(testDataView);

                if (calibrated)
                {
                    CalibratedBinaryClassificationMetrics metrics = _mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label");
                    ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
                    //Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

                }
                else
                {
                    BinaryClassificationMetrics metrics = _mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, labelColumnName: "Label");
                    ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
                    //Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

                }

            //Save the model file that can be used by any application
            if (_baseModelsRelativePath != null)
                {
                    string modelRelativeLocation = $"{_baseModelsRelativePath}/{trainer.name}Model.zip";
                    string modelPath = FileHelpers.GetAbsolutePath(modelRelativeLocation);
                    _mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
                    ConsoleHelper.WriteLog($"The model is saved to {modelPath}");
                }

                #region "STEP 5:  Order features by importance"

                

                #endregion
            
        }

        public void GenerateTrainersModel(IDataView trainingDataView, IDataView testDataView,
            (string name, IEstimator<ITransformer> value) trainer,
            EstimatorChain<NormalizingTransformer> pipelineBase,
            bool calibrated = false)
        {
            //Train and test generators

            var pipeline = pipelineBase
                            .Append(trainer.value);

            // 3. Phase for Training, Evaluation and model file persistence
            // Per each regression trainer: Train, Evaluate, and Save a different model

            ConsoleHelper.WriteLog($"=============== Training the {trainer} model ===============");
            var trainedModel = pipeline.Fit(trainingDataView);

            ConsoleHelper.WriteLog($"===== Evaluating {trainer} Model's accuracy with Test data =====");
            IDataView predictions = trainedModel.Transform(testDataView);

            if (calibrated)
            {
                CalibratedBinaryClassificationMetrics metrics = _mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Label");
                ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
                Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            }
            else
            {
                BinaryClassificationMetrics metrics = _mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, labelColumnName: "Label");
                ConsoleHelper.PrintBinaryClassificationMetrics(trainer.ToString(), metrics);
                Console.WriteLine(metrics.ConfusionMatrix.GetFormattedConfusionTable());

            }

            var a = trainer.value.Preview(predictions, 10, 10);
            //Save the model file that can be used by any application
            if (_baseModelsRelativePath != null)
            {
                string modelRelativeLocation = $"{_baseModelsRelativePath}/{trainer.name}Model.zip";
                string modelPath = FileHelpers.GetAbsolutePath(modelRelativeLocation);
                _mlContext.Model.Save(trainedModel, trainingDataView.Schema, modelPath);
                ConsoleHelper.WriteLog($"The model is saved to {modelPath}");
            }

            #region "STEP 5:  Order features by importance"



            #endregion

        }

        private void OrderFeaturesByImportance(string name, IDataView trainingDataView)
        {
            switch (name)
            {
                case "SdcaLogisticRegression":
                   OrderFeaturesByImportance_SdcaLogisticRegression(trainingDataView);break;
                case "FastTree": OrderFeaturesByImportance_FastTree(trainingDataView); break;
                case "AveragedPerceptron": OrderFeaturesByImportance_AveragedPerceptron(trainingDataView); break;
                case "FastForest": OrderFeaturesByImportance_FastForest(trainingDataView); break;
                case "SdcaNonCalibrated": OrderFeaturesByImportance_SdcaNonCalibrated(trainingDataView); break;
                case "LbfgsLogisticRegression": OrderFeaturesByImportance_LbfgsLogisticRegression(trainingDataView); break;
                case "LinearSvm": LinearSvm(trainingDataView); break;
                case "Prior": OrderFeaturesByImportance_Prior(trainingDataView); break;
                case "SgdNonCalibrated": OrderFeaturesByImportance_SgdNonCalibrated(trainingDataView); break;
                case "SgdCalibrated": OrderFeaturesByImportance_SgdCalibrated(trainingDataView); break;
            }
        }

        private void OrderFeaturesByImportance_SgdCalibrated(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .SgdCalibrated(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void OrderFeaturesByImportance_SgdNonCalibrated(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .SgdNonCalibrated(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void OrderFeaturesByImportance_Prior(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .SgdCalibrated(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void LinearSvm(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .LinearSvm(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void OrderFeaturesByImportance_LbfgsLogisticRegression(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .LbfgsLogisticRegression(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void OrderFeaturesByImportance_SdcaNonCalibrated(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .SdcaNonCalibrated(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void OrderFeaturesByImportance_FastForest(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .FastForest(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void OrderFeaturesByImportance_AveragedPerceptron(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .AveragedPerceptron(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.Ranking
                .PermutationFeatureImportance(linearPredictor, transformedData, labelColumnName: "Label", permutationCount: 30);


            var sortedIndices = permutationMetrics.Select((metrics, index) => new
            {
                index,
                metrics.NormalizedDiscountedCumulativeGains
            })
                .OrderByDescending(feature => Math.Abs(
                    feature.NormalizedDiscountedCumulativeGains[0].Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Change in NDCG@1#95% Confidence in the" +
                "Mean Change in NDCG@1");
            var ndcg = permutationMetrics.Select(
                x => x.NormalizedDiscountedCumulativeGains).ToArray();
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:G4}#{2:G4}",
                    i,
                    ndcg[i][0].Mean,
                    1.96 * ndcg[i][0].StandardError);
            }
        }

        private void OrderFeaturesByImportance_FastTree(IDataView data)
        {
            //var featureColumns =
            //    new string[] { nameof(data.Feature1), nameof(Data.Feature2) };
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .FastTree(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.BinaryClassification.PermutationFeatureImportance(linearPredictor, transformedData, permutationCount: 30, labelColumnName: "Label");

            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on AUC.
            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.AreaUnderRocCurve })
                .OrderByDescending(
                feature => Math.Abs(feature.AreaUnderRocCurve.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Model Weight#Change in AUC"
                + "#95% Confidence in the Mean Change in AUC");
            var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray();
            foreach (int i in sortedIndices)
            {
                //Console.WriteLine("{0}#{1:0.00}#{2:G4}#{3:G4}",
                //    featureColumns[i],
                //    linearPredictor.Model.SubModel.Weights[i],
                //    auc[i].Mean,
                //    1.96 * auc[i].StandardError);
            }
        }

        private List<string> GetColumnName(IDataView data)
        {
            List<string> featureColumns = new List<string>();
            foreach (DataViewSchema.Column item in data.Schema)
            {
                featureColumns.Add(item.Name);
            }
            return featureColumns;
        }
        private void OrderFeaturesByImportance_SdcaLogisticRegression(IDataView data)
        {
            // Define a training pipeline that concatenates features into a vector,
            // normalizes them, and then trains a linear model.
            var pipeline = FeaturizeEstimator()
                            .Append(_mlContext.BinaryClassification.Trainers
                            .SdcaLogisticRegression(labelColumnName: "Label"));

            // Fit the pipeline to the data.
            var model = pipeline.Fit(data);

            // Transform the dataset.
            var transformedData = model.Transform(data);

            // Extract the predictor.
            var linearPredictor = model.LastTransformer;

            // Compute the permutation metrics for the linear model using the
            // normalized data.
            var permutationMetrics = _mlContext.BinaryClassification.PermutationFeatureImportance(linearPredictor, transformedData, permutationCount: 30, labelColumnName: "Label");

            // Now let's look at which features are most important to the model
            // overall. Get the feature indices sorted by their impact on AUC.
            var sortedIndices = permutationMetrics
                .Select((metrics, index) => new { index, metrics.Accuracy })
                .OrderByDescending(
                feature => Math.Abs(feature.Accuracy.Mean))
                .Select(feature => feature.index);

            Console.WriteLine("Feature#Model Weight#Change in AUC"
                + "#95% Confidence in the Mean Change in AUC #FMS");
            var auc = permutationMetrics.Select(x => x.AreaUnderRocCurve).ToArray();

            var columnName = GetColumnName(data);
            foreach (int i in sortedIndices)
            {
                Console.WriteLine("{0}#{1:0.00}#{2:G4}#{3:G4}",
                    columnName[i],
                    linearPredictor.Model.SubModel.Weights [i],
                    auc[i].Mean,
                    1.96 * auc[i].StandardError);
            }
        }





        /// Generate Trainers Model
        /// </summary>
        /// <param name="dataView"></param>
        /// <param name="binaryClassificationLearners"></param>
        public string  GenerateTrainersModelCrossValidation(IDataView dataView, (string name, IEstimator<ITransformer> value) binaryClassificationLearner, EstimatorChain<NormalizingTransformer> dataProcessPipeline)
        {
            string modelPath = "";
            //Microsoft.ML.Trainers.FastTree.FastTreeTweedieTrainer trainer = _mlContext.Regression.Trainers.FastTreeTweedie(labelColumnName: "Label", featureColumnName: "Features");

            // Cross-Validate with single dataset
            ConsoleHelper.WriteLog($"=============== Cross-validating to get {binaryClassificationLearner.name} model's accuracy metrics ===============");
            //var crossValidationResults = _mlContext.Regression.CrossValidate(data: dataView, estimator: trainingPipeline, numberOfFolds: 6, labelColumnName: "Label");
            //ConsoleHelper.PrintRegressionFoldsAverageMetrics(trainer.ToString(), crossValidationResults);

            TransformerChain<ITransformer> trainedModel = null;
            // 3. Phase for Training, Evaluation and model file persistence
            // Train, Evaluate, and Save a different model
            {
                Console.WriteLine($"=============== Cross validation {binaryClassificationLearner.name} trainer ===============");
                var trainingPipeline = dataProcessPipeline.Append(binaryClassificationLearner.value);
                if (binaryClassificationLearner.IsCalibrated())
                {
                    IReadOnlyList<TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics>> crossValidationResults = _mlContext.BinaryClassification.CrossValidate(data: dataView, estimator: trainingPipeline, numberOfFolds: 10, labelColumnName: "Label");

                    ConsoleHelper.PrintBinaryClassificationFoldsAverageMetrics(binaryClassificationLearner.name, crossValidationResults);

                    trainedModel = trainingPipeline.Fit(dataView);
                }
                else
                {
                    IReadOnlyList<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> crossValidationResults = _mlContext.BinaryClassification.CrossValidateNonCalibrated(data: dataView, estimator: trainingPipeline, numberOfFolds: 10, labelColumnName: "Label");
                    ConsoleHelper.PrintNonCalibratedBinaryClassificationFoldsAverageMetrics(binaryClassificationLearner.name, crossValidationResults);

                    trainedModel = trainingPipeline.Fit(dataView);

                    //Console.WriteLine(crossValidationResults.First().Metrics.ConfusionMatrix.GetFormattedConfusionTable());

                    //ConsoleHelper.WriteLog("===== Evaluating Model's accuracy with Test data =====");
                    //IDataView predictions = trainedModel.Transform(dataView);
                    //BinaryClassificationMetrics metrics = _mlContext.BinaryClassification.EvaluateNonCalibrated(data: predictions, labelColumnName: "Label");
                    //ConsoleHelper.PrintBinaryClassificationMetrics(binaryClassificationLearner.name, metrics);
                }

                ////Save the model file that can be used by any application
                //if (_baseModelsRelativePath != null)
                //{
                //    string modelRelativeLocation = $"{_baseModelsRelativePath}/{binaryClassificationLearner.name}Model.zip";
                //    modelPath = FileHelpers.GetAbsolutePath(modelRelativeLocation);
                //    _mlContext.Model.Save(trainedModel, dataView.Schema, modelPath);
                //    Console.WriteLine($"The model is saved to {modelPath}");
                //}
            }
            return modelPath;
        }

        public void EvaluateModel((string name, IEstimator<ITransformer> value) estimator, string validationSetPath)
        {
            // 4. Try/test Predictions with the created models
            // The following test predictions could be implemented/deployed in a different application (production apps)
            // that's why it is seggregated from the previous loop
            // For each trained model, test 10 predictions           
            //foreach ((string name, IEstimator<ITransformer> _) in binaryClassificationLearner)
            {
                //Load current model from .ZIP file
                string modelRelativeLocation = $"{_baseModelsRelativePath}/{estimator.name}Model.zip";
                ITransformer trainedModel = _mlContext.Model.Load(modelRelativeLocation, out var modelInputSchema);

                // Create prediction engine related to the loaded trained model
                var predEngine = _mlContext.Model.CreatePredictionEngine<BankruptcyObservation, BankruptcyPrediction>(trainedModel);

                ConsoleHelper.WriteLog($"================== Visualize data for model {estimator.name}Model.zip ==================");
                //Visualize data test dataset
                //ModelScoringTester.VisualizeSomePredictions(_mlContext, learner.name, $"{_locationPathForPersistence}/data/{learner.name}T1_12_ balanced.csv", predEngine, 0);
                ModelScoringTester.VisualizeSomePredictions(_mlContext, estimator.name, validationSetPath , predEngine, 0);
            }
        }

        public void EvaluateModel(string name, ITransformer trainedModel)
        {
            // 4. Try/test Predictions with the created models
            // The following test predictions could be implemented/deployed in a different application (production apps)
            // that's why it is seggregated from the previous loop
            // For each trained model, test 10 predictions           
            //foreach ((string name, IEstimator<ITransformer> _) in binaryClassificationLearner)
            {
                // Create prediction engine related to the loaded trained model
                var predEngine = _mlContext.Model.CreatePredictionEngine<BankruptcyObservation, BankruptcyPrediction>(trainedModel);

                ConsoleHelper.WriteLog($"================== Visualize data for model {name}Model.zip ==================");
                //Visualize data test dataset
                //ModelScoringTester.VisualizeSomePredictions(_mlContext, learner.name, $"{_locationPathForPersistence}/data/{learner.name}T1_12_ balanced.csv", predEngine, 0);
                ModelScoringTester.VisualizeSomePredictions(_mlContext, name, $"{_baseDatasetsRelativePath}/T1_12_ balanced.csv", predEngine, 0);
            }
        }

    }
}
