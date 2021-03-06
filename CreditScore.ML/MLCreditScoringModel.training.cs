﻿// This file was auto-generated by ML.NET Model Builder. 
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;
using Microsoft.ML.Trainers.FastTree;
using Microsoft.ML.Trainers;
using Microsoft.ML;

namespace CreditScore_ML
{
    public partial class MLCreditScoringModel
    {
        public static ITransformer RetrainPipeline(MLContext context, IDataView trainData)
        {
            var pipeline = BuildPipeline(context);
            var model = pipeline.Fit(trainData);

            return model;
        }

        /// <summary>
        /// build the pipeline that is used from model builder. Use this function to retrain model.
        /// </summary>
        /// <param name="mlContext"></param>
        /// <returns></returns>
        public static IEstimator<ITransformer> BuildPipeline(MLContext mlContext)
        {
            // Data process configuration with pipeline data transformations
            var pipeline = mlContext.Transforms.ReplaceMissingValues(new []{new InputOutputColumnPair(@"V1", @"V1"),new InputOutputColumnPair(@"V2", @"V2"),new InputOutputColumnPair(@"V3", @"V3"),new InputOutputColumnPair(@"V4", @"V4"),new InputOutputColumnPair(@"V5", @"V5"),new InputOutputColumnPair(@"V6", @"V6"),new InputOutputColumnPair(@"V7", @"V7"),new InputOutputColumnPair(@"V8", @"V8"),new InputOutputColumnPair(@"V9", @"V9"),new InputOutputColumnPair(@"V10", @"V10"),new InputOutputColumnPair(@"V11", @"V11"),new InputOutputColumnPair(@"V12", @"V12"),new InputOutputColumnPair(@"V13", @"V13"),new InputOutputColumnPair(@"V14", @"V14")})      
                                    .Append(mlContext.Transforms.Concatenate(@"Features", new []{@"V1",@"V2",@"V3",@"V4",@"V5",@"V6",@"V7",@"V8",@"V9",@"V10",@"V11",@"V12",@"V13",@"V14"}))      
                                    .Append(mlContext.Transforms.Conversion.MapValueToKey(@"Fracaso", @"Fracaso"))      
                                    .Append(mlContext.MulticlassClassification.Trainers.OneVersusAll(binaryEstimator:mlContext.BinaryClassification.Trainers.FastForest(new FastForestBinaryTrainer.Options(){NumberOfTrees=4,FeatureFraction=1F,LabelColumnName=@"Fracaso",FeatureColumnName=@"Features"}), labelColumnName: @"Fracaso"))      
                                    .Append(mlContext.Transforms.Conversion.MapKeyToValue(@"PredictedLabel", @"PredictedLabel"));

            return pipeline;
        }
    }
}
