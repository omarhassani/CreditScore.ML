using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CreditScore.ML.Extensions
{
    public static class DiccionaryExtensions
    {
        public static bool IsCalibrated(this (string name, IEstimator<ITransformer> valuje) instance)
        {
            switch(instance.name)
            {
                case "FastTree": return true;
               case "SdcaNonCalibrated": return false;
                case "FieldAwareFactorizationMachine": return true;
                case "Gam": return true;
                case "LbfgsLogisticRegression": return true;
                case "SdcaLogisticRegression": return true;
                case "Prior": return true;
                case "SgdCalibrated": return true;

                case "AveragedPerceptron": return false;
                case "FastForest": return false;
                
                case "LdSvm": return false;
                case "LinearSvm": return false;
                case "SgdNonCalibrated": return false;
                default: return false;
            }
        }
    }
}
