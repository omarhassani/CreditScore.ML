using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CreditScore.ML.Model
{
    public class BankruptcyPrediction
    {
        [ColumnName("Label")]
        public bool Bankruptcy;
    }
}
