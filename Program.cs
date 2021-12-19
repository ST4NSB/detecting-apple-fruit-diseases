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
            dl.TrainModel(trainingImages,
                         (msg) => Console.WriteLine(msg),
                         modelArch: Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.ResnetV250);

            Console.WriteLine("\n");
            var results = dl.TestModel(testImages);
            dl.Evaluate(results, (msg) => Console.Write(msg), showConfusionMatrix: false);

            Console.WriteLine("\n");
            var sampleImages = Helpers.GetSampleImages();
            var sampleResults = dl.TestModel(sampleImages);
            Console.WriteLine(dl.ShowResults(sampleResults));
        }
    }
}
