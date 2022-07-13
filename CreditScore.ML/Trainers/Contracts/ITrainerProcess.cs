using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CreditScore.ML.Trainers.Contracts
{
    public interface ITrainerProcess
    {
        void Process(IDataView trainingSetPath, IDataView validationSetPath);
    }
}
