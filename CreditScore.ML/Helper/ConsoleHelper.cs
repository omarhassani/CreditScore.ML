using Microsoft.ML;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static Microsoft.ML.TrainCatalogBase;

namespace CreditScore.ML
{
    public static class ConsoleHelper
    {
        static readonly string _modelPath = @"../../../../Logs/Log.txt";
        public static void WriteLog(string stringText)
        {
            Console.WriteLine(stringText);
            // Create a StreamWriter from FileStream  
            using (StreamWriter writer = new StreamWriter(_modelPath))
            {
                writer.Write(stringText);
            }
        }
        public static void WriteLog(string format, params object?[]? arg)
        {
            Console.WriteLine(format, arg);
            // Create a StreamWriter from FileStream  
            using (StreamWriter writer = new StreamWriter(_modelPath))
            {
                writer.Write(format, arg);
            }
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(string clasifierName, IReadOnlyList<CrossValidationResult<MulticlassClassificationMetrics>> crossValidationResults)
        {
            var metricsInMultipleFolds = crossValidationResults.Select(r => r.Metrics);

            var microAccuracyValues = metricsInMultipleFolds.Select(m => m.MicroAccuracy);
            var microAccuracyAverage = microAccuracyValues.Average();
            var microAccuraciesStdDeviation = CalculateStandardDeviation(microAccuracyValues);
            var microAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(microAccuracyValues);

            var macroAccuracyValues = metricsInMultipleFolds.Select(m => m.MacroAccuracy);
            var macroAccuracyAverage = macroAccuracyValues.Average();
            var macroAccuraciesStdDeviation = CalculateStandardDeviation(macroAccuracyValues);
            var macroAccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(macroAccuracyValues);

            var logLossValues = metricsInMultipleFolds.Select(m => m.LogLoss);
            var logLossAverage = logLossValues.Average();
            var logLossStdDeviation = CalculateStandardDeviation(logLossValues);
            var logLossConfidenceInterval95 = CalculateConfidenceInterval95(logLossValues);

            var logLossReductionValues = metricsInMultipleFolds.Select(m => m.LogLossReduction);
            var logLossReductionAverage = logLossReductionValues.Average();
            var logLossReductionStdDeviation = CalculateStandardDeviation(logLossReductionValues);
            var logLossReductionConfidenceInterval95 = CalculateConfidenceInterval95(logLossReductionValues);

            WriteLog($"*************************************************************************************************************");
            WriteLog($"*       Metrics for {clasifierName} Multi-class Classification model      ");
            WriteLog($"*------------------------------------------------------------------------------------------------------------");
            WriteLog($"*       Average MicroAccuracy:    {microAccuracyAverage:0.###}  - Standard deviation: ({microAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({microAccuraciesConfidenceInterval95:#.###})");
            WriteLog($"*       Average MacroAccuracy:    {macroAccuracyAverage:0.###}  - Standard deviation: ({macroAccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({macroAccuraciesConfidenceInterval95:#.###})");
            WriteLog($"*       Average LogLoss:          {logLossAverage:#.###}  - Standard deviation: ({logLossStdDeviation:#.###})  - Confidence Interval 95%: ({logLossConfidenceInterval95:#.###})");
            WriteLog($"*       Average LogLossReduction: {logLossReductionAverage:#.###}  - Standard deviation: ({logLossReductionStdDeviation:#.###})  - Confidence Interval 95%: ({logLossReductionConfidenceInterval95:#.###})");
            WriteLog($"*************************************************************************************************************");
        }

        public static void PrintMulticlassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            WriteLog($"************************************************************");
            WriteLog($"*       Metrics for {name} Multi class classification model      ");
            WriteLog($"*-----------------------------------------------------------");
            WriteLog($"*       MicroAccuracy: {metrics.MicroAccuracy:P2}");
            WriteLog($"*       MacroAccuracy:      {metrics.MacroAccuracy :P2}");
            WriteLog($"*       LogLoss:  {metrics.LogLoss:P2}");
            WriteLog($"*       LogLossReduction:  {metrics.LogLossReduction:P2}");
            WriteLog($"*       TopKAccuracy:  {metrics.TopKAccuracy:#.##}");
            WriteLog($"*       TopKAccuracyForAllK:  {metrics.TopKAccuracyForAllK:#.##}");
            WriteLog($"*       TopKPredictionCount:  {metrics.TopKPredictionCount:#.##}");
            WriteLog($"*       ConfusionMatrix:  {Newtonsoft.Json.JsonConvert.SerializeObject(metrics.ConfusionMatrix)}");
            WriteLog($"*       TopKPredictionCount:  {metrics.TopKPredictionCount}");
            WriteLog($"************************************************************");
        }

        public static void PrintPrediction(string prediction)
        {
            WriteLog($"*************************************************");
            WriteLog($"Predicted : {prediction}");
            WriteLog($"*************************************************");
        }

        public static void PrintRegressionPredictionVersusObserved(string predictionCount, string observedCount)
        {
            WriteLog($"-------------------------------------------------");
            WriteLog($"Predicted : {predictionCount}");
            WriteLog($"Actual:     {observedCount}");
            WriteLog($"-------------------------------------------------");
        }

        public static void PrintRegressionMetrics(string name, RegressionMetrics metrics)
        {
            WriteLog($"*************************************************");
            WriteLog($"*       Metrics for {name} regression model      ");
            WriteLog($"*------------------------------------------------");
            WriteLog($"*       LossFn:        {metrics.LossFunction:0.##}");
            WriteLog($"*       R2 Score:      {metrics.RSquared:0.##}");
            WriteLog($"*       Absolute loss: {metrics.MeanAbsoluteError:#.##}");
            WriteLog($"*       Squared loss:  {metrics.MeanSquaredError:#.##}");
            WriteLog($"*       RMS loss:      {metrics.RootMeanSquaredError:#.##}");
            WriteLog($"*************************************************");
        }

        public static void PrintBinaryClassificationMetrics(string name, BinaryClassificationMetrics metrics)
        {
            WriteLog($"************************************************************");
            WriteLog($"*       Metrics for {name} binary classification model      ");
            WriteLog($"*-----------------------------------------------------------");
            WriteLog($"*       Accuracy: {metrics.Accuracy:P2}");
            WriteLog($"*       Area Under Curve:      {metrics.AreaUnderRocCurve:P2}");
            WriteLog($"*       Area under Precision recall Curve:  {metrics.AreaUnderPrecisionRecallCurve:P2}");
            WriteLog($"*       F1Score:  {metrics.F1Score:P2}");
            WriteLog($"*       PositivePrecision:  {metrics.PositivePrecision:#.##}");
            WriteLog($"*       PositiveRecall:  {metrics.PositiveRecall:#.##}");
            WriteLog($"*       NegativePrecision:  {metrics.NegativePrecision:#.##}");
            WriteLog($"*       NegativeRecall:  {metrics.NegativeRecall:P2}");
            WriteLog($"*       ConfusionMatrix:  {Newtonsoft.Json.JsonConvert.SerializeObject(metrics.ConfusionMatrix)}");
            WriteLog($"*       F1Score:  {metrics.F1Score}");
            WriteLog($"************************************************************");
        }


        
        public static void PrintBinaryClassificationFoldsAverageMetrics(
                                 string algorithmName,
                               IReadOnlyList<TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics>> crossValResults
                                                                   )
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);
            var AccuracyAverage = AccuracyValues.Average();
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);

            var AreaUnderRocCurveValues = metricsInMultipleFolds.Select(m => m.AreaUnderRocCurve);
            var AreaUnderRocCurveAverage = AreaUnderRocCurveValues.Average();
            var AreaUnderRocCurveStdDeviation = CalculateStandardDeviation(AreaUnderRocCurveValues);
            var AreaUnderRocCurveConfidenceInterval95 = CalculateConfidenceInterval95(AreaUnderRocCurveValues);

            var AreaUnderPrecisionRecallCurveValues = metricsInMultipleFolds.Select(m => m.AreaUnderPrecisionRecallCurve);
            var AreaUnderPrecisionRecallCurveAverage = AreaUnderPrecisionRecallCurveValues.Average();
            var AreaUnderPrecisionRecallCurveStdDeviation = CalculateStandardDeviation(AreaUnderPrecisionRecallCurveValues);
            var AreaUnderPrecisionRecallCurveConfidenceInterval95 = CalculateConfidenceInterval95(AreaUnderPrecisionRecallCurveValues);

            var F1ScoreValues = metricsInMultipleFolds.Select(m => m.F1Score);
            var F1ScoreAverage = F1ScoreValues.Average();
            var F1ScoreStdDeviation = CalculateStandardDeviation(F1ScoreValues);
            var F1ScoreConfidenceInterval95 = CalculateConfidenceInterval95(F1ScoreValues);

            var PositiveRecallValues = metricsInMultipleFolds.Select(m => m.PositiveRecall);
            var PositiveRecallAverage = PositiveRecallValues.Average();
            var PositiveRecallStdDeviation = CalculateStandardDeviation(PositiveRecallValues);
            var PositiveRecallConfidenceInterval95 = CalculateConfidenceInterval95(PositiveRecallValues);

            var NegativePrecisionValues = metricsInMultipleFolds.Select(m => m.NegativePrecision);
            var NegativePrecisionAverage = NegativePrecisionValues.Average();
            var NegativePrecisionStdDeviation = CalculateStandardDeviation(NegativePrecisionValues);
            var NegativePrecisionConfidenceInterval95 = CalculateConfidenceInterval95(NegativePrecisionValues);

            var PositivePrecisionValues = metricsInMultipleFolds.Select(m => m.PositivePrecision);
            var PositivePrecisionAverage = PositivePrecisionValues.Average();
            var PositivePrecisionStdDeviation = CalculateStandardDeviation(PositivePrecisionValues);
            var PositivePrecisionConfidenceInterval95 = CalculateConfidenceInterval95(PositivePrecisionValues);

            var NegativeRecallValues = metricsInMultipleFolds.Select(m => m.NegativeRecall);
            var NegativeRecallAverage = NegativeRecallValues.Average();
            var NegativeRecallStdDeviation = CalculateStandardDeviation(NegativeRecallValues);
            var NegativeRecallConfidenceInterval95 = CalculateConfidenceInterval95(NegativeRecallValues);


            WriteLog($"*       Metrics for {algorithmName} binary classification model      ");
            WriteLog($"Accuracy - {AccuracyAverage:0.###}");
            WriteLog($"Area Under Roc Curve Accuracy - {AreaUnderRocCurveAverage:#.###}");
            WriteLog($"Area Under PRC - {AreaUnderPrecisionRecallCurveAverage:#.###}");
            WriteLog($"F1Score - {F1ScoreAverage:#.###}");
            WriteLog($"Negative Precision - {NegativePrecisionAverage:#.###}");
            WriteLog($"Positive Precision - {PositivePrecisionAverage:#.###}");
            WriteLog($"Positive Recall - {PositiveRecallAverage:#.###}");
            WriteLog($"Negative Recall - {NegativeRecallAverage:#.###}");


            //WriteLog($"*       Metrics for {algorithmName} binary classification model      ");
            //WriteLog($"Metric - Average - Standard deviation - Confidence Interval 95%");
            //WriteLog($"Accuracy - {AccuracyAverage:0.###} - {AccuraciesStdDeviation:#.###} - {AccuraciesConfidenceInterval95:#.###}");
            //WriteLog($"Area Under Roc Curve Accuracy - {AreaUnderRocCurveAverage:#.###} - {AreaUnderRocCurveStdDeviation:#.###} - {AreaUnderRocCurveConfidenceInterval95:#.###}");
            //WriteLog($"Area Under PRC - {AreaUnderPrecisionRecallCurveAverage:#.###} - {AreaUnderPrecisionRecallCurveStdDeviation:#.###} - {AreaUnderPrecisionRecallCurveConfidenceInterval95:#.###}");
            //WriteLog($"F1Score - {F1ScoreAverage:#.###} - {F1ScoreStdDeviation:#.###} - {F1ScoreConfidenceInterval95:#.###}");
            //WriteLog($"Negative Precision - {NegativePrecisionAverage:#.###} - {NegativePrecisionStdDeviation:#.###} - {NegativePrecisionConfidenceInterval95:#.###}");
            //WriteLog($"Positive Precision - {PositivePrecisionAverage:#.###} - {PositivePrecisionStdDeviation:#.###} - {PositivePrecisionConfidenceInterval95:#.###}");
            //WriteLog($"Positive Recall - {PositiveRecallAverage:#.###} - {PositiveRecallStdDeviation:#.###} - {PositiveRecallConfidenceInterval95:#.###}");
            //WriteLog($"Negative Recall - {NegativeRecallAverage:#.###} - {NegativeRecallStdDeviation:#.###} - {NegativeRecallConfidenceInterval95:#.###}");
            //foreach (TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics> item in crossValResults)
            //{
            //    Console.WriteLine(item.Metrics.ConfusionMatrix.GetFormattedConfusionTable());
            //}
        }

        public static void PrintMulticlassClassificationFoldsAverageMetrics(
                         string algorithmName,
                       IReadOnlyList<TrainCatalogBase.CrossValidationResult<CalibratedBinaryClassificationMetrics>> crossValResults
                                                           )
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);
            var AccuracyAverage = AccuracyValues.Average();
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);

            var AreaUnderRocCurveValues = metricsInMultipleFolds.Select(m => m.AreaUnderRocCurve);
            var AreaUnderRocCurveAverage = AreaUnderRocCurveValues.Average();
            var AreaUnderRocCurveStdDeviation = CalculateStandardDeviation(AreaUnderRocCurveValues);
            var AreaUnderRocCurveConfidenceInterval95 = CalculateConfidenceInterval95(AreaUnderRocCurveValues);

            var AreaUnderPrecisionRecallCurveValues = metricsInMultipleFolds.Select(m => m.AreaUnderPrecisionRecallCurve);
            var AreaUnderPrecisionRecallCurveAverage = AreaUnderPrecisionRecallCurveValues.Average();
            var AreaUnderPrecisionRecallCurveStdDeviation = CalculateStandardDeviation(AreaUnderPrecisionRecallCurveValues);
            var AreaUnderPrecisionRecallCurveConfidenceInterval95 = CalculateConfidenceInterval95(AreaUnderPrecisionRecallCurveValues);

            var F1ScoreValues = metricsInMultipleFolds.Select(m => m.F1Score);
            var F1ScoreAverage = F1ScoreValues.Average();
            var F1ScoreStdDeviation = CalculateStandardDeviation(F1ScoreValues);
            var F1ScoreConfidenceInterval95 = CalculateConfidenceInterval95(F1ScoreValues);

            var PositiveRecallValues = metricsInMultipleFolds.Select(m => m.PositiveRecall);
            var PositiveRecallAverage = PositiveRecallValues.Average();
            var PositiveRecallStdDeviation = CalculateStandardDeviation(PositiveRecallValues);
            var PositiveRecallConfidenceInterval95 = CalculateConfidenceInterval95(PositiveRecallValues);

            var NegativePrecisionValues = metricsInMultipleFolds.Select(m => m.NegativePrecision);
            var NegativePrecisionAverage = NegativePrecisionValues.Average();
            var NegativePrecisionStdDeviation = CalculateStandardDeviation(NegativePrecisionValues);
            var NegativePrecisionConfidenceInterval95 = CalculateConfidenceInterval95(NegativePrecisionValues);

            var PositivePrecisionValues = metricsInMultipleFolds.Select(m => m.PositivePrecision);
            var PositivePrecisionAverage = PositivePrecisionValues.Average();
            var PositivePrecisionStdDeviation = CalculateStandardDeviation(PositivePrecisionValues);
            var PositivePrecisionConfidenceInterval95 = CalculateConfidenceInterval95(PositivePrecisionValues);

            var NegativeRecallValues = metricsInMultipleFolds.Select(m => m.NegativeRecall);
            var NegativeRecallAverage = NegativeRecallValues.Average();
            var NegativeRecallStdDeviation = CalculateStandardDeviation(NegativeRecallValues);
            var NegativeRecallConfidenceInterval95 = CalculateConfidenceInterval95(NegativeRecallValues);

            //WriteLog($"*************************************************************************************************************");
            //WriteLog($"*       Metrics for {algorithmName} binary Classification model      ");
            //WriteLog($"*------------------------------------------------------------------------------------------------------------");
            //WriteLog($"*       Average Accuracy:    {AccuracyAverage:0.###}  - Standard deviation: ({AccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({AccuraciesConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Area Under Roc Curve:          {AreaUnderRocCurveAverage:#.###}  - Standard deviation: ({AreaUnderRocCurveStdDeviation:#.###})  - Confidence Interval 95%: ({AreaUnderRocCurveConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Area Under Precision Recall Curve:          {AreaUnderPrecisionRecallCurveAverage:#.###}  - Standard deviation: ({AreaUnderPrecisionRecallCurveStdDeviation:#.###})  - Confidence Interval 95%: ({AreaUnderPrecisionRecallCurveConfidenceInterval95:#.###})");

            //WriteLog($"*       Average F1Score: {F1ScoreAverage:#.###}  - Standard deviation: ({F1ScoreStdDeviation:#.###})  - Confidence Interval 95%: ({F1ScoreConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Negative Precision: {NegativePrecisionAverage:#.###}  - Standard deviation: ({NegativePrecisionStdDeviation:#.###})  - Confidence Interval 95%: ({NegativePrecisionConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Positive Precision: {PositivePrecisionAverage:#.###}  - Standard deviation: ({PositivePrecisionStdDeviation:#.###})  - Confidence Interval 95%: ({PositivePrecisionConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Positive Recall: {PositiveRecallAverage:#.###}  - Standard deviation: ({PositiveRecallStdDeviation:#.###})  - Confidence Interval 95%: ({PositiveRecallConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Negative Recall: {NegativeRecallAverage:#.###}  - Standard deviation: ({NegativeRecallStdDeviation:#.###})  - Confidence Interval 95%: ({NegativeRecallConfidenceInterval95:#.###})");
            //WriteLog($"*************************************************************************************************************");
            //--------------
            //WriteLog($"*************************************************************************************************************");
            //WriteLog($"Metrics for {algorithmName} binary Classification model      ");
            //WriteLog($"-----------------------------------------------------------------------------------------------------");
            //WriteLog($"Metric - Average - Standard deviation - Confidence Interval 95%");
            //WriteLog($"Accuracy - {AccuracyAverage:0.###} - {AccuraciesStdDeviation:#.###} - {AccuraciesConfidenceInterval95:#.###}");
            //WriteLog($"Area Under Roc Curve Accuracy - {AreaUnderRocCurveAverage:#.###} - {AreaUnderRocCurveStdDeviation:#.###} - {AreaUnderRocCurveConfidenceInterval95:#.###}");
            //WriteLog($"Area Under PRC - {AreaUnderPrecisionRecallCurveAverage:#.###} - {AreaUnderPrecisionRecallCurveStdDeviation:#.###} - {AreaUnderPrecisionRecallCurveConfidenceInterval95:#.###}");
            //WriteLog($"F1Score - {F1ScoreAverage:#.###} - {F1ScoreStdDeviation:#.###} - {F1ScoreConfidenceInterval95:#.###}");
            //WriteLog($"Negative Precision - {NegativePrecisionAverage:#.###} - {NegativePrecisionStdDeviation:#.###} - {NegativePrecisionConfidenceInterval95:#.###}");
            //WriteLog($"Positive Precision - {PositivePrecisionAverage:#.###} - {PositivePrecisionStdDeviation:#.###} - {PositivePrecisionConfidenceInterval95:#.###}");
            //WriteLog($"Positive Recall - {PositiveRecallAverage:#.###} - {PositiveRecallStdDeviation:#.###} - {PositiveRecallConfidenceInterval95:#.###}");
            //WriteLog($"Negative Recall - {NegativeRecallAverage:#.###} - {NegativeRecallStdDeviation:#.###} - {NegativeRecallConfidenceInterval95:#.###}");


            WriteLog($"*       Metrics for {algorithmName} binary classification model      ");
            WriteLog($"Accuracy - {AccuracyAverage:0.###}");
            WriteLog($"Area Under Roc Curve - {AreaUnderRocCurveAverage:#.###}");
            WriteLog($"Area Under PRC - {AreaUnderPrecisionRecallCurveAverage:#.###}");
            WriteLog($"F1Score - {F1ScoreAverage:#.###}");
            WriteLog($"Negative Precision - {NegativePrecisionAverage:#.###}");
            WriteLog($"Positive Precision - {PositivePrecisionAverage:#.###}");
            WriteLog($"Positive Recall - {PositiveRecallAverage:#.###}");
            WriteLog($"Negative Recall - {NegativeRecallAverage:#.###}");

        }

        public static void PrintNonCalibratedBinaryClassificationFoldsAverageMetrics(
                         string algorithmName,
                       IReadOnlyList<TrainCatalogBase.CrossValidationResult<BinaryClassificationMetrics>> crossValResults
                                                           )
        {
            var metricsInMultipleFolds = crossValResults.Select(r => r.Metrics);

            var AccuracyValues = metricsInMultipleFolds.Select(m => m.Accuracy);
            var AccuracyAverage = AccuracyValues.Average();
            var AccuraciesStdDeviation = CalculateStandardDeviation(AccuracyValues);
            var AccuraciesConfidenceInterval95 = CalculateConfidenceInterval95(AccuracyValues);

            var AreaUnderRocCurveValues = metricsInMultipleFolds.Select(m => m.AreaUnderRocCurve);
            var AreaUnderRocCurveAverage = AreaUnderRocCurveValues.Average();
            var AreaUnderRocCurveStdDeviation = CalculateStandardDeviation(AreaUnderRocCurveValues);
            var AreaUnderRocCurveConfidenceInterval95 = CalculateConfidenceInterval95(AreaUnderRocCurveValues);

            var AreaUnderPrecisionRecallCurveValues = metricsInMultipleFolds.Select(m => m.AreaUnderPrecisionRecallCurve);
            var AreaUnderPrecisionRecallCurveAverage = AreaUnderPrecisionRecallCurveValues.Average();
            var AreaUnderPrecisionRecallCurveStdDeviation = CalculateStandardDeviation(AreaUnderPrecisionRecallCurveValues);
            var AreaUnderPrecisionRecallCurveConfidenceInterval95 = CalculateConfidenceInterval95(AreaUnderRocCurveValues);

            var F1ScoreValues = metricsInMultipleFolds.Select(m => m.F1Score);
            var F1ScoreAverage = F1ScoreValues.Average();
            var F1ScoreStdDeviation = CalculateStandardDeviation(F1ScoreValues);
            var F1ScoreConfidenceInterval95 = CalculateConfidenceInterval95(F1ScoreValues);

            var PositiveRecallValues = metricsInMultipleFolds.Select(m => m.PositiveRecall);
            var PositiveRecallAverage = PositiveRecallValues.Average();
            var PositiveRecallStdDeviation = CalculateStandardDeviation(PositiveRecallValues);
            var PositiveRecallConfidenceInterval95 = CalculateConfidenceInterval95(PositiveRecallValues);

            var NegativePrecisionValues = metricsInMultipleFolds.Select(m => m.NegativePrecision);
            var NegativePrecisionAverage = NegativePrecisionValues.Average();
            var NegativePrecisionStdDeviation = CalculateStandardDeviation(NegativePrecisionValues);
            var NegativePrecisionConfidenceInterval95 = CalculateConfidenceInterval95(NegativePrecisionValues);

            var PositivePrecisionValues = metricsInMultipleFolds.Select(m => m.PositivePrecision);
            var PositivePrecisionAverage = PositivePrecisionValues.Average();
            var PositivePrecisionStdDeviation = CalculateStandardDeviation(PositivePrecisionValues);
            var PositivePrecisionConfidenceInterval95 = CalculateConfidenceInterval95(PositivePrecisionValues);

            var NegativeRecallValues = metricsInMultipleFolds.Select(m => m.NegativeRecall);
            var NegativeRecallAverage = NegativeRecallValues.Average();
            var NegativeRecallStdDeviation = CalculateStandardDeviation(NegativeRecallValues);
            var NegativeRecallConfidenceInterval95 = CalculateConfidenceInterval95(NegativeRecallValues);

            //WriteLog($"*************************************************************************************************************");
            //WriteLog($"*       Metrics for {algorithmName} binary Classification model      ");
            //WriteLog($"*------------------------------------------------------------------------------------------------------------");
            //WriteLog($"*       Average Accuracy:    {AccuracyAverage:0.###}  - Standard deviation: ({AccuraciesStdDeviation:#.###})  - Confidence Interval 95%: ({AccuraciesConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Area Under Roc Curve:          {AreaUnderRocCurveAverage:#.###}  - Standard deviation: ({AreaUnderRocCurveStdDeviation:#.###})  - Confidence Interval 95%: ({AreaUnderRocCurveConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Area Under Precision Recall Curve:          {AreaUnderPrecisionRecallCurveAverage:#.###}  - Standard deviation: ({AreaUnderPrecisionRecallCurveStdDeviation:#.###})  - Confidence Interval 95%: ({AreaUnderPrecisionRecallCurveConfidenceInterval95:#.###})");

            //WriteLog($"*       Average F1Score: {F1ScoreAverage:#.###}  - Standard deviation: ({F1ScoreStdDeviation:#.###})  - Confidence Interval 95%: ({F1ScoreConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Negative Precision: {NegativePrecisionAverage:#.###}  - Standard deviation: ({NegativePrecisionStdDeviation:#.###})  - Confidence Interval 95%: ({NegativePrecisionConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Positive Precision: {PositivePrecisionAverage:#.###}  - Standard deviation: ({PositivePrecisionStdDeviation:#.###})  - Confidence Interval 95%: ({PositivePrecisionConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Positive Recall: {PositiveRecallAverage:#.###}  - Standard deviation: ({PositiveRecallStdDeviation:#.###})  - Confidence Interval 95%: ({PositiveRecallConfidenceInterval95:#.###})");
            //WriteLog($"*       Average Negative Recall: {NegativeRecallAverage:#.###}  - Standard deviation: ({NegativeRecallStdDeviation:#.###})  - Confidence Interval 95%: ({NegativeRecallConfidenceInterval95:#.###})");
            //WriteLog($"*************************************************************************************************************");
            //WriteLog($"");
            //WriteLog($"Metrics for {algorithmName} binary Classification model      ");
            //WriteLog($"");
            //WriteLog($"Metric - Average - Standard deviation - Confidence Interval 95%");
            //WriteLog($"Accuracy - {AccuracyAverage:0.###} - {AccuraciesStdDeviation:#.###} - {AccuraciesConfidenceInterval95:#.###}");
            //WriteLog($"Area Under Roc Curve Accuracy - {AreaUnderRocCurveAverage:#.###} - {AreaUnderRocCurveStdDeviation:#.###} - {AreaUnderRocCurveConfidenceInterval95:#.###}");
            //WriteLog($"Area Under PRC - {AreaUnderPrecisionRecallCurveAverage:#.###} - {AreaUnderPrecisionRecallCurveStdDeviation:#.###} - {AreaUnderPrecisionRecallCurveConfidenceInterval95:#.###}");
            //WriteLog($"F1Score - {F1ScoreAverage:#.###} - {F1ScoreStdDeviation:#.###} - {F1ScoreConfidenceInterval95:#.###}");
            //WriteLog($"Negative Precision - {NegativePrecisionAverage:#.###} - {NegativePrecisionStdDeviation:#.###} - {NegativePrecisionConfidenceInterval95:#.###}");
            //WriteLog($"Positive Precision - {PositivePrecisionAverage:#.###} - {PositivePrecisionStdDeviation:#.###} - {PositivePrecisionConfidenceInterval95:#.###}");
            //WriteLog($"Positive Recall - {PositiveRecallAverage:#.###} - {PositiveRecallStdDeviation:#.###} - {PositiveRecallConfidenceInterval95:#.###}");
            //WriteLog($"Negative Recall - {NegativeRecallAverage:#.###} - {NegativeRecallStdDeviation:#.###} - {NegativeRecallConfidenceInterval95:#.###}");
            WriteLog($"*       Metrics for {algorithmName} binary classification model      ");
            WriteLog($"Accuracy - {AccuracyAverage:0.###}");
            WriteLog($"Area Under Roc - {AreaUnderRocCurveAverage:#.###}");
            WriteLog($"Area Under PRC - {AreaUnderPrecisionRecallCurveAverage:#.###}");
            WriteLog($"F1Score - {F1ScoreAverage:#.###}");
            WriteLog($"Negative Precision - {NegativePrecisionAverage:#.###}");
            WriteLog($"Positive Precision - {PositivePrecisionAverage:#.###}");
            WriteLog($"Positive Recall - {PositiveRecallAverage:#.###}");
            WriteLog($"Negative Recall - {NegativeRecallAverage:#.###}");

        }
        public static void PrintAnomalyDetectionMetrics(string name, AnomalyDetectionMetrics metrics)
        {
            WriteLog($"************************************************************");
            WriteLog($"*       Metrics for {name} anomaly detection model      ");
            WriteLog($"*-----------------------------------------------------------");
            WriteLog($"*       Area Under ROC Curve:                       {metrics.AreaUnderRocCurve:P2}");
            WriteLog($"*       Detection rate at false positive count: {metrics.DetectionRateAtFalsePositiveCount}");
            WriteLog($"************************************************************");
        }

        public static void PrintMultiClassClassificationMetrics(string name, MulticlassClassificationMetrics metrics)
        {
            WriteLog($"************************************************************");
            WriteLog($"*    Metrics for {name} multi-class classification model   ");
            WriteLog($"*-----------------------------------------------------------");
            WriteLog($"    AccuracyMacro = {metrics.MacroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            WriteLog($"    AccuracyMicro = {metrics.MicroAccuracy:0.####}, a value between 0 and 1, the closer to 1, the better");
            WriteLog($"    LogLoss = {metrics.LogLoss:0.####}, the closer to 0, the better");
            WriteLog($"    LogLoss for class 1 = {metrics.PerClassLogLoss[0]:0.####}, the closer to 0, the better");
            WriteLog($"    LogLoss for class 2 = {metrics.PerClassLogLoss[1]:0.####}, the closer to 0, the better");
            WriteLog($"    LogLoss for class 3 = {metrics.PerClassLogLoss[2]:0.####}, the closer to 0, the better");
            WriteLog($"************************************************************");
        }

        public static void PrintRegressionFoldsAverageMetrics(string algorithmName, IReadOnlyList<CrossValidationResult<RegressionMetrics>> crossValidationResults)
        {
            var L1 = crossValidationResults.Select(r => r.Metrics.MeanAbsoluteError);
            var L2 = crossValidationResults.Select(r => r.Metrics.MeanSquaredError);
            var RMS = crossValidationResults.Select(r => r.Metrics.RootMeanSquaredError);
            var lossFunction = crossValidationResults.Select(r => r.Metrics.LossFunction);
            var R2 = crossValidationResults.Select(r => r.Metrics.RSquared);

            WriteLog($"*************************************************************************************************************");
            WriteLog($"*       Metrics for {algorithmName} Regression model      ");
            WriteLog($"*------------------------------------------------------------------------------------------------------------");
            WriteLog($"*       Average L1 Loss:    {L1.Average():0.###} ");
            WriteLog($"*       Average L2 Loss:    {L2.Average():0.###}  ");
            WriteLog($"*       Average RMS:          {RMS.Average():0.###}  ");
            WriteLog($"*       Average Loss Function: {lossFunction.Average():0.###}  ");
            WriteLog($"*       Average R-squared: {R2.Average():0.###}  ");
            WriteLog($"*************************************************************************************************************");
        }



        public static double CalculateStandardDeviation(IEnumerable<double> values)
        {
            double average = values.Average();
            double sumOfSquaresOfDifferences = values.Select(val => (val - average) * (val - average)).Sum();
            double standardDeviation = Math.Sqrt(sumOfSquaresOfDifferences / (values.Count() - 1));
            return standardDeviation;
        }

        public static double CalculateConfidenceInterval95(IEnumerable<double> values)
        {
            double confidenceInterval95 = 1.96 * CalculateStandardDeviation(values) / Math.Sqrt((values.Count() - 1));
            return confidenceInterval95;
        }

        public static void PrintClusteringMetrics(string name, ClusteringMetrics metrics)
        {
            WriteLog($"*************************************************");
            WriteLog($"*       Metrics for {name} clustering model      ");
            WriteLog($"*------------------------------------------------");
            WriteLog($"*       Average Distance: {metrics.AverageDistance}");
            WriteLog($"*       Davies Bouldin Index is: {metrics.DaviesBouldinIndex}");
            WriteLog($"*************************************************");
        }

        public static void ShowDataViewInConsole(MLContext mlContext, IDataView dataView, int numberOfRows = 4)
        {
            string msg = string.Format("Show data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            var preViewTransformedData = dataView.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                WriteLog(lineToPrint + "\n");
            }
        }

        [Conditional("DEBUG")]
        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekDataViewInConsole(MLContext mlContext, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: Showing {0} rows with the columns", numberOfRows.ToString());
            ConsoleWriteHeader(msg);

            //https://github.com/dotnet/machinelearning/blob/main/docs/code/MlNetCookBook.md#how-do-i-look-at-the-intermediate-data
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // 'transformedData' is a 'promise' of data, lazy-loading. call Preview
            //and iterate through the returned collection from preview.

            var preViewTransformedData = transformedData.Preview(maxRows: numberOfRows);

            foreach (var row in preViewTransformedData.RowView)
            {
                var ColumnCollection = row.Values;
                string lineToPrint = "Row--> ";
                foreach (KeyValuePair<string, object> column in ColumnCollection)
                {
                    lineToPrint += $"| {column.Key}:{column.Value}";
                }
                WriteLog(lineToPrint + "\n");
            }
        }

        [Conditional("DEBUG")]
        // This method using 'DebuggerExtensions.Preview()' should only be used when debugging/developing, not for release/production trainings
        public static void PeekVectorColumnDataInConsole(MLContext mlContext, string columnName, IDataView dataView, IEstimator<ITransformer> pipeline, int numberOfRows = 4)
        {
            string msg = string.Format("Peek data in DataView: : Show {0} rows with just the '{1}' column", numberOfRows, columnName);
            ConsoleWriteHeader(msg);

            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // Extract the 'Features' column.
            var someColumnData = transformedData.GetColumn<float[]>(columnName)
                                                        .Take(numberOfRows).ToList();

            // print to console the peeked rows

            int currentRow = 0;
            someColumnData.ForEach(row => {
                currentRow++;
                String concatColumn = String.Empty;
                foreach (float f in row)
                {
                    concatColumn += f.ToString();
                }

                string rowMsg = string.Format("**** Row {0} with '{1}' field value ****", currentRow, columnName);
                WriteLog(rowMsg);
                WriteLog(concatColumn);

            });
        }

        public static void ConsoleWriteHeader(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Yellow;
            WriteLog(" ");
            foreach (var line in lines)
            {
                WriteLog(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            WriteLog(new string('#', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsoleWriterSection(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Blue;
            WriteLog(" ");
            foreach (var line in lines)
            {
                WriteLog(line);
            }
            var maxLength = lines.Select(x => x.Length).Max();
            WriteLog(new string('-', maxLength));
            Console.ForegroundColor = defaultColor;
        }

        public static void ConsolePressAnyKey()
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Green;
            WriteLog(" ");
            WriteLog("Press any key to finish.");
            Console.ReadKey();
        }

        public static void ConsoleWriteException(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.Red;
            const string exceptionTitle = "EXCEPTION";
            WriteLog(" ");
            WriteLog(exceptionTitle);
            WriteLog(new string('#', exceptionTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                WriteLog(line);
            }
        }

        public static void ConsoleWriteWarning(params string[] lines)
        {
            var defaultColor = Console.ForegroundColor;
            Console.ForegroundColor = ConsoleColor.DarkMagenta;
            const string warningTitle = "WARNING";
            WriteLog(" ");
            WriteLog(warningTitle);
            WriteLog(new string('#', warningTitle.Length));
            Console.ForegroundColor = defaultColor;
            foreach (var line in lines)
            {
                WriteLog(line);
            }
        }

       
    }

    
}
