using Microsoft.ML;
using Microsoft.ML.Transforms;
using Microsoft.ML.Data;
using Microsoft.ML.Vision;
using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using static Microsoft.ML.DataOperationsCatalog;
using static Microsoft.ML.Vision.ImageClassificationTrainer;

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
                               bool shuffle = true, 
                               double validationSplit = 0.2, 
                               Architecture modelArch = Architecture.ResnetV250,
                               int epoch = 100,
                               int batchSize = 20,
                               float learningRate = 0.01f)
        {
            var imageDataView = _ctx.Data.LoadFromEnumerable(inputData);

            if (shuffle)
            {
                imageDataView = _ctx.Data.ShuffleRows(imageDataView);
            }

            var trainValidationData = _ctx.Data.TrainTestSplit(imageDataView, testFraction: validationSplit);
            var (trainSet, validationSet) = CreateTrainingSets(trainValidationData);
            
            _predictionModel = CreateModel(modelArch, (trainSet, validationSet), epoch, batchSize, learningRate);
            _predictionEngine = _ctx.Model.CreatePredictionEngine<ImageModelInput, ImagePrediction>(_predictionModel);
        }

        public IEnumerable<ImagePrediction> TestModel(IEnumerable<ImageModelInput> inputData)
        {
            var testDataView = _ctx.Data.LoadFromEnumerable(inputData);
            var testSet = CreateTestSet(testDataView);

            var predictions = _predictionModel.Transform(testSet);
            return _ctx.Data.CreateEnumerable<ImagePrediction>(predictions, reuseRowObject: false);
        }

        public string PredictSingleImage(ImageModelInput image)
        {
            var output = new ImagePrediction();
            _predictionEngine.Predict(image, ref output);
            return output.PredictedLabel;
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
                ValidationSet = sets.validationSet
            };

            var pipeline = _ctx.MulticlassClassification.Trainers.ImageClassification(options)
                .Append(_ctx.Transforms.Conversion.MapKeyToValue("PredictedLabel"));

            return pipeline.Fit(sets.trainSet);
        }
    }
}
