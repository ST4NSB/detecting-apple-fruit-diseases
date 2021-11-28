using System;

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
                         modelArch: Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.MobilenetV2,
                         epoch: 3, 
                         batchSize: 5);
            var results = dl.TestModel(testImages);
            dl.Evaluate(results, (msg) => Console.Write(msg));
        }
    }
}
