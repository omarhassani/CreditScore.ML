using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.DataOperationsCatalog;

namespace CreditScore.ML
{
    public static class old
    {
        private static readonly string BaseDatasetsRelativePath = @"../../../../Data";
        private static readonly string DataRelativePath = $@"C:\Users\omar.hassani\Documents\Personal\PFM\CsvData\T1.cs";

        private static readonly string DataPath = GetAbsolutePath(DataRelativePath);

        private static readonly string BaseModelsRelativePath = @"../../../../MLModels";

        private static readonly string ModelRelativePath = $"{BaseModelsRelativePath}/CreditScoring.zip";
        private static readonly string ModelPath = GetAbsolutePath(ModelRelativePath);
        public static void OldMain(string[] args)
        {
            // Create MLContext to be shared across the model creation workflow objects 
            // Set a random seed for repeatable/deterministic results across multiple trainings.
            var mlContext = new MLContext(seed: 1);

            // STEP 1: Common data loading configuration
            IDataView dataView = mlContext.Data.LoadFromTextFile<CreditScoringModel>(@"C:\Users\omar.hassani\Documents\Personal\PFM\CsvData\T1.csv", ';', true);

            TrainTestData trainTestSplit = mlContext.Data.TrainTestSplit(dataView, testFraction: 0.2);
            IDataView trainingData = trainTestSplit.TrainSet;
            IDataView testData = trainTestSplit.TestSet;

            // STEP 2: Common data process configuration with pipeline data transformations          
            TextFeaturizingEstimator dataProcessPipeline = mlContext.Transforms.Text.FeaturizeText(outputColumnName: "Features", inputColumnName: nameof(CreditScoringModel.CNAE));

            // STEP 3.1: Set the training algorithm, then create and config the modelBuilder                            
            var sdcaLogisticRegressionTrainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(labelColumnName: "Bankruptcy", featureColumnName: "Features", null, null, null, 1000);
            var sdcaLogisticRegressionTrainingPipeline = dataProcessPipeline.Append(sdcaLogisticRegressionTrainer);

            // STEP 4.1: Train the model fitting to the DataSet
            ITransformer trainedModel = sdcaLogisticRegressionTrainingPipeline.Fit(trainingData);

            //// STEP 3.1: Set the training algorithm, then create and config the modelBuilder                            
            //var trainer = mlContext.BinaryClassification.Trainers.AveragedPerceptron(labelColumnName: "Quiebra", featureColumnName: "Features");
            //var trainingPipeline = dataProcessPipeline.Append(sdcaLogisticRegressionTrainer);

            //// STEP 4.1: Train the model fitting to the DataSet
            //ITransformer trainedModel = sdcaLogisticRegressionTrainingPipeline.Fit(trainingData);


            // STEP 5: Evaluate the model and show accuracy stats
            var predictions = trainedModel.Transform(testData);
            var metrics = mlContext.BinaryClassification.Evaluate(data: predictions, labelColumnName: "Bankruptcy", scoreColumnName: "Score");

            ConsoleHelper.PrintBinaryClassificationMetrics(sdcaLogisticRegressionTrainer.ToString(), metrics);

            // STEP 6: Save/persist the trained model to a .ZIP file
            //mlContext.Model.Save(trainedModel, trainingData.Schema, ModelPath);

            //Console.WriteLine("The model is saved to {0}", ModelPath);

            //// TRY IT: Make a single test prediction, loading the model from .ZIP file
            //// Create single instance of sample data from first line of dataset for model input
            //MLCreditScoringModel.ModelInput sampleData = new MLCreditScoringModel.ModelInput()
            //{
            //    V1 = 3272882F,
            //    V2 = 1.42152E+09F,
            //    V3 = 2.006667E+07F,
            //    V4 = 2.333762E+08F,
            //    V5 = 2.189443E+07F,
            //    V6 = 7.475295E+08F,
            //    V7 = 5.969168E+08F,
            //    V8 = 2.664508E+08F,
            //    V9 = 6.767217E+07F,
            //    V10 = 1.271051E+07F,
            //    V11 = 7.425057E+08F,
            //    V12 = 6659963F,
            //    V13 = -5.316088E+07F,
            //    CNAE = "sasas",
            //};

            //// Make a single prediction on the sample data and print results
            //var predictionResult = MLCreditScoringModel.Predict(sampleData);

            //Console.WriteLine("Using model to make single prediction -- Comparing actual Quiebra with predicted Quiebra from sample data...\n\n");


            //Console.WriteLine($"V1: {3272882F}");
            //Console.WriteLine($"V2: {1.42152E+09F}");
            //Console.WriteLine($"V3: {2.006667E+07F}");
            //Console.WriteLine($"V4: {2.333762E+08F}");
            //Console.WriteLine($"V5: {2.189443E+07F}");
            //Console.WriteLine($"V6: {7.475295E+08F}");
            //Console.WriteLine($"V7: {5.969168E+08F}");
            //Console.WriteLine($"V8: {2.664508E+08F}");
            //Console.WriteLine($"V9: {6.767217E+07F}");
            //Console.WriteLine($"V10: {1.271051E+07F}");
            //Console.WriteLine($"V11: {7.425057E+08F}");
            //Console.WriteLine($"V12: {6659963F}");
            //Console.WriteLine($"V13: {-5.316088E+07F}");
            //Console.WriteLine($"CNAE: {27F}");
            //Console.WriteLine($"Quiebra: {0F}");


            //Console.WriteLine($"\n\nPredicted Quiebra: {predictionResult.Prediction}\n\n");
            //Console.WriteLine("=============== End of process, hit any key to finish ===============");
            //Console.ReadKey();


        }

        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
    }
}
