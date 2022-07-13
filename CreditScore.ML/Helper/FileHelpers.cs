using Microsoft.ML;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using static CreditScore_ML.MLCreditScoringModel;

namespace CreditScore.ML.Helper
{
    public static class FileHelpers
    {
        public static string GetAbsolutePath(string relativePath)
        {
            FileInfo _dataRoot = new FileInfo(typeof(Program).Assembly.Location);
            string assemblyFolderPath = _dataRoot.Directory.FullName;

            string fullPath = Path.Combine(assemblyFolderPath, relativePath);

            return fullPath;
        }
        public static List<CreditScoringModel> ReadCsv(string fileName)
        {
            var reader = new StreamReader(File.OpenRead(fileName));
            List<CreditScoringModel> CreditScoringModelList = new();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (!line.StartsWith("V1"))
                {
                    var values = line.Split(',');

                    CreditScoringModelList.Add(new CreditScoringModel(
                        float.Parse(values[0]),
                        float.Parse(values[1]),
                        float.Parse(values[2]),
                        float.Parse(values[3]),
                        float.Parse(values[4]),
                        float.Parse(values[5]),
                        float.Parse(values[6]),
                        float.Parse(values[7]),
                        float.Parse(values[8]),
                        float.Parse(values[9]),
                        float.Parse(values[10]),
                        float.Parse(values[11]),
                        float.Parse(values[12]),
                        float.Parse(values[13]),
                       Convert.ToBoolean(Convert.ToInt32(values[14]))));

                    //multiClasifier ? Convert.ToBoolean(Convert.ToInt32(values[14])) : Convert.ToUInt32(values[14])));
                }
            }

            return CreditScoringModelList;
        }

        public static List<ModelInput> ReadCsvModelInput(string fileName)
        {
            var reader = new StreamReader(File.OpenRead(fileName));
            List<ModelInput> CreditScoringModelList = new();
            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (!line.StartsWith("V1"))
                {
                    var values = line.Split(';');

                    CreditScoringModelList.Add(new ModelInput()
                    {
                        V1 = float.Parse(values[0]),
                        V2 = float.Parse(values[1]),
                        V3 = float.Parse(values[2]),
                        V4 = float.Parse(values[3]),
                        V5 = float.Parse(values[4]),
                        V6 = float.Parse(values[5]),
                        V7 = float.Parse(values[6]),
                        V8 = float.Parse(values[7]),
                        V9 = float.Parse(values[8]),
                        V10 = float.Parse(values[9]),
                        V11 = float.Parse(values[10]),
                        V12 = float.Parse(values[11]),
                        V13 = float.Parse(values[12]),
                        V14 = float.Parse(values[13]),
                        Fracaso = Convert.ToInt32(values[14])
                    });
                }
            }

            return CreditScoringModelList;
        }

        public static List<string> GetColumnNameExcept_Label_Cnae(IDataView data)
        {
            List<string> featureColumns = new List<string>();
            foreach (DataViewSchema.Column item in data.Schema.Where(c => c.Name != "Label" && c.Name != "V14" && c.Name != "Fracaso"))
            {
                featureColumns.Add(item.Name);
            }
            return featureColumns;
        }

        private static List<ModelInput> GenerateData(string trainingSetPath )
        {
            return ReadCsvModelInput(trainingSetPath);
        }


    }
}
