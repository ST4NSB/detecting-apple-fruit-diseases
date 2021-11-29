using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections.Generic;
using static Microsoft.ML.DataOperationsCatalog;
using static Microsoft.ML.Vision.ImageClassificationTrainer;
using System.Diagnostics;
using System.Text;
using System.Linq;

namespace DetectingAppleDiseases
{
    public class DeepLearning
    {
        private MLContext _ctx;
        private readonly string _trainImagesPath, _testImagesPath;
        private TransformerChain<KeyToValueMappingTransformer> _predictionModel;
        private PredictionEngine<ImageModelInput, ImagePrediction> _predictionEngine;

        public DeepLearning(string trainImagesPath, string testImagesPath)
        {
            _trainImagesPath = trainImagesPath;
            _testImagesPath = testImagesPath;
            _ctx = new MLContext();
        }

        public void TrainModel(IEnumerable<ImageData> inputData, 
                               Action<string> log,
                               Architecture modelArch,
                               bool shuffle = true,
                               double validationSplit = 0.25,
                               int epoch = 100,
                               int batchSize = 64,
                               float learningRate = 0.1f)
        {
            var stopWatch = new Stopwatch();
            stopWatch.Start();
            var imageDataView = _ctx.Data.LoadFromEnumerable(inputData);

            if (shuffle) imageDataView = _ctx.Data.ShuffleRows(imageDataView);
            
            var trainValidationData = _ctx.Data.TrainTestSplit(imageDataView, testFraction: validationSplit);
            var (trainSet, validationSet) = CreateTrainingSets(trainValidationData);
            
            _predictionModel = CreateModel(modelArch, (trainSet, validationSet), epoch, batchSize, learningRate);
            _predictionEngine = _ctx.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(_predictionModel);
            
            stopWatch.Stop();
            var ts = stopWatch.Elapsed;

            string elapsedTime = String.Format("{0:00}:{1:00}:{2:00}.{3:00} (Hours:Minutes:Seconds.Milliseconds)", 
                                                ts.Hours, ts.Minutes, ts.Seconds, ts.Milliseconds / 10);
            log("Training model runTime: " + elapsedTime);
        }

        public IDataView TestModel(IEnumerable<ImageModelInput> inputData)
        {
            var testDataView = _ctx.Data.LoadFromEnumerable(inputData);
            var testSet = CreateTestSet(testDataView);

            return _predictionModel.Transform(testSet);
        }

        public void Evaluate(IDataView predDataView, 
                             Action<string> log, 
                             bool showLogLoss = true, 
                             bool showAccuracy = true,
                             bool showConfusionMatrix = true)
        {
            var predEnum = _ctx.Data.CreateEnumerable<ImagePrediction>(predDataView, reuseRowObject: false);

            if (showLogLoss)
            {
                var metrics = _ctx.MulticlassClassification.Evaluate(predDataView, labelColumnName: "LabelKey", predictedLabelColumnName: "PredictedLabel");
                log($"LogLoss: {metrics.LogLoss}\n");
            }

            if (showAccuracy)
            {
                log($"Accuracy: {Evaluation.GetNaiveAccuracy(predEnum)}\n");
            }

            if (showConfusionMatrix)
            {
                var eval = new Evaluation();
                eval.CreateSupervizedEvaluationsMatrix(predEnum);

                log("TAG\t\t\tACCURACY\t\t\tPRECISION\t\t\tRECALL(TPR)\t\t\tSPECIFICITY(TNR)\t\t\tF1-SCORE\n");
                var fullMatrix = eval.PrintClassificationResultsMatrix();
                for (int i = 0; i < eval.GetFullMatrixLineLength(); i++)
                {
                    for (int j = 0; j < eval.GetFullMatrixColLength(); j++)
                        log(fullMatrix[i][j] + "\t\t\t");
                    log("\n");
                }
            }
        }

        public string ShowResults(IDataView resultsDataView)
        {
            var predEnum = _ctx.Data.CreateEnumerable<ImagePrediction>(resultsDataView, reuseRowObject: false);
            var results = new StringBuilder();

            foreach(var item in predEnum)
            {
                var name = item.Name.Split("\\").Last();
                results.Append($"Prediction for '{name}': {item.PredictedLabel}\n");
            }

            return results.ToString();
        }

        private (IDataView trainSet, IDataView validationSet) CreateTrainingSets(TrainTestData trainValidationData)
        {
            var validationSet = _ctx.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(_ctx.Transforms.LoadRawImageBytes("Image", _trainImagesPath, "Name"))
                .Fit(trainValidationData.TestSet)
                .Transform(trainValidationData.TestSet);

            var trainSet = _ctx.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(_ctx.Transforms.LoadRawImageBytes("Image", _trainImagesPath, "Name"))
                .Fit(trainValidationData.TrainSet)
                .Transform(trainValidationData.TrainSet);

            return (trainSet, validationSet);
        }

        private IDataView CreateTestSet(IDataView testDataView)
        {
            var testSet = _ctx.Transforms.Conversion.MapValueToKey("LabelKey", "Label", keyOrdinality: ValueToKeyMappingEstimator.KeyOrdinality.ByValue)
                .Append(_ctx.Transforms.LoadRawImageBytes("Image", _testImagesPath, "Name"))
                .Fit(testDataView)
                .Transform(testDataView);

            return testSet;
        }

        private TransformerChain<KeyToValueMappingTransformer> CreateModel(Architecture arch, (IDataView trainSet, IDataView validationSet) sets, int epoch, int batchSize, float learningRate)
        {
            var options = new ImageClassificationTrainer.Options()
            {
                Arch = arch,
                Epoch = epoch,
                BatchSize = batchSize,
                LearningRate = learningRate,
                LabelColumnName = "LabelKey",
                FeatureColumnName = "Image",
                MetricsCallback = (metrics) => Console.WriteLine(metrics),
                ValidationSet = sets.validationSet
            };

            var pipeline = _ctx.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(_ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            return pipeline.Fit(sets.trainSet);
        }
    }
}
