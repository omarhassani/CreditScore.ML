using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CreditScore.ML.Trainers.Helpers
{
    public class FeatureHelper
    {
        public static List<string> GetColumnName(IDataView data)
        {
            List<string> featureColumns = new List<string>();
            foreach (DataViewSchema.Column item in data.Schema.Where(c => c.Name != "Label"))
            {
                featureColumns.Add(item.Name);
            }
            return featureColumns;
        }

    }
}
