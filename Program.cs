using System;
using System.Linq;

namespace DetectingAppleDiseases
{
    class Program
    {
        static void Main(string[] args)
        {
            var trainPath = Helpers.GetDatasetImagesPath(evaluationType: "Train");
            var trainingImages = Helpers.GetTrainingImages();
            
            var testPath = Helpers.GetDatasetImagesPath(evaluationType: "Test");
            var testImages = Helpers.GetTestImages();
            
            DeepLearning dl = new DeepLearning(trainPath, testPath);
            dl.TrainModel(trainingImages, epoch: 3, batchSize: 5);
            var results = dl.TestModel(testImages);

            var eval = new Evaluation();
            Console.WriteLine($"Accuracy: {Evaluation.GetNaiveAccuracy(results)}");
            eval.CreateSupervizedEvaluationsMatrix(results);
            
            Console.WriteLine("TAG\t\tACCURACY\t\tPRECISION\t\tRECALL(TPR)\t\tSPECIFICITY(TNR)\t\tF1-SCORE");
            var fullMatrix = eval.PrintClassificationResultsMatrix();
            for (int i = 0; i < eval.GetFullMatrixLineLength(); i++)
            {
                for (int j = 0; j < eval.GetFullMatrixColLength(); j++)
                    Console.Write(fullMatrix[i][j] + "\t\t");
                Console.WriteLine();
            }

        }
    }
}
