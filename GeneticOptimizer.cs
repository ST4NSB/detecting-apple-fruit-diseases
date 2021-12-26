using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DetectingAppleDiseases
{
    public class GeneticOptimizer
    {
        private const int _seed = 42;
        private const int _epochMin = 1, _epochMax = 101; // 1 - 100
        private const int _batchSizeMin = 10, _batchSizeMax = 101; // 10 - 100 
        private const int _learningRateMin = 1, _learningRateMax = 10; // 1 - 9

        private const float _fitnessThreshold = 8.25f;

        private int _population, _selection, _generationTotalNumbers;
        private IEnumerable<ImageData> _trainImages;
        private IEnumerable<ImageModelInput> _testImages;
        private string _testPath, _trainPath;

        private struct ChromosomeInformation
        {
            public int Epochs;
            public int BatchSize;
            public float LearningRate;
            public float Fitness;

            public ChromosomeInformation(int epochs, int batchSize, float learningRate, float fitness = float.MaxValue)
            {
                Epochs = epochs;
                BatchSize = batchSize;
                LearningRate = learningRate;
                Fitness = fitness;
            }
        }

        public GeneticOptimizer(int population = 12, int selection = 6, int totalNumberOfGenerations = 15)
        {
            _population = population;
            _selection = selection;
            _generationTotalNumbers = totalNumberOfGenerations;
        }

        public void GenerateWorld()
        {
            _trainPath = Helpers.GetDatasetImagesPath(evaluationType: "Train");
            _trainImages = Helpers.GetTrainingImages();

            _testPath = Helpers.GetDatasetImagesPath(evaluationType: "Test");
            _testImages = Helpers.GetTestImages();

            Console.WriteLine(" *** Wait a few minutes .. This make take a while..");
            var chromosomes = new List<ChromosomeInformation>();
            for (int i = 0; i < _population; i++)
            {
                var chromosome = GenerateRandomChromosome();
                var fitness = GetFitnessValue(chromosome);
                chromosomes.Add(new ChromosomeInformation(chromosome.epochs, 
                                                          chromosome.batchSize, 
                                                          chromosome.learningRate, 
                                                          fitness));
                PrintInformations(chromosomes.Last());
            }

            chromosomes = chromosomes.OrderByDescending(x => x.Fitness).ToList();

            var generation = 1;
            while (generation <= _generationTotalNumbers)
            {
                Console.WriteLine(" ~~~~~");
                Console.WriteLine($" Generation number: {generation++}");
                PrintInformations(chromosomes[0]);

                var childs = ApplyCrossover(chromosomes);
                Console.WriteLine(" Done crossover");
                var mutatedChilds = MutateChilds(childs);
                Console.WriteLine(" ~~~~~");
                chromosomes = new List<ChromosomeInformation>(mutatedChilds);
                chromosomes = chromosomes.OrderByDescending(x => x.Fitness).ToList();
                chromosomes = chromosomes.Take(_selection).ToList();

                if (chromosomes.First().Fitness >= _fitnessThreshold) break;
            }

            PrintInformations(chromosomes[0]);
        }

        private void PrintInformations(ChromosomeInformation chromosome)
        {
            Console.WriteLine(" *** Chromosome configuration ***");
            Console.WriteLine($" Number of epochs: {chromosome.Epochs}");
            Console.WriteLine($" Batch size: {chromosome.BatchSize}");
            Console.WriteLine($" Learning rate: {chromosome.LearningRate}");
            Console.WriteLine($" ! Fitness (Accuracy): {chromosome.Fitness}");
            Console.WriteLine(" *** ------------ ***");
        }

        private List<ChromosomeInformation> ApplyCrossover(List<ChromosomeInformation> parents)
        {
            var rand = new Random();
            var childs = new List<ChromosomeInformation>();

            for (int i = 0; i < parents.Count; i += 2)
            {
                var epochMin = Math.Min(parents[i].Epochs, parents[i + 1].Epochs);
                var epochMax = Math.Max(parents[i].Epochs, parents[i + 1].Epochs);

                var batchSizeMin = Math.Min(parents[i].BatchSize, parents[i + 1].BatchSize);
                var batchSizeMax = Math.Max(parents[i].BatchSize, parents[i + 1].BatchSize);


                var epochsIn = rand.Next(epochMin, epochMax + 1);
                var batchSizeIn = rand.Next(batchSizeMin, batchSizeMax + 1);

                var lrFirst = GetLearningRateIntervalNumber(parents[i].LearningRate);
                var lrSecond = GetLearningRateIntervalNumber(parents[i+1].LearningRate);

                var learningRateMin = Math.Min(lrFirst, lrSecond) == 0 ? _learningRateMin : Math.Min(lrFirst, lrSecond);
                var learningRateMax = Math.Max(lrFirst, lrSecond) == 0 ? _learningRateMax : Math.Max(lrFirst, lrSecond);
                var learningRateIn = rand.Next(learningRateMin, learningRateMax + 1);

                childs.Add(new ChromosomeInformation(epochsIn, batchSizeIn, (float)Math.Pow(10, 0 - learningRateIn)));

                var prePostInterval = rand.Next(2); // pre - 0, post - 1
                var epochOut = prePostInterval == 0 ? rand.Next(_epochMin, epochMin) : rand.Next(epochMax, _epochMax);
                prePostInterval = rand.Next(2);
                var batchSizeOut = prePostInterval == 0 ? rand.Next(_batchSizeMin, batchSizeMin) : rand.Next(batchSizeMax, _batchSizeMax);
                prePostInterval = rand.Next(2);
                Console.WriteLine($" Learning rate (min, max): {learningRateMin}, {learningRateMax}");
                var learningRateOut = prePostInterval == 0 ? rand.Next(_learningRateMin, learningRateMin) : rand.Next(learningRateMax, _learningRateMax);

                childs.Add(new ChromosomeInformation(epochOut, batchSizeOut, (float)Math.Pow(10, 0 - learningRateOut)));
            }

            Console.WriteLine($" Crossover completed! Number of children: {childs.Count}");

            return childs;
        }

        private List<ChromosomeInformation> MutateChilds(List<ChromosomeInformation> childs)
        {
            var rand = new Random();
            var mutatedChilds = new List<ChromosomeInformation>();

            foreach (var child in childs)
            {
                var newEpochs = child.Epochs;
                var newBatchSize = child.BatchSize;
                var newLearningRate = child.LearningRate;
               
                var mutationConfig = rand.Next(6);
                switch(mutationConfig)
                {
                    case 0:
                        newEpochs = rand.Next(_epochMin, _epochMax);
                        break;
                    case 1:
                        newBatchSize = rand.Next(_batchSizeMin, _batchSizeMax);
                        break;
                    case 2:
                        newLearningRate = (float)Math.Pow(10, 0 - rand.Next(_learningRateMin, _learningRateMax));
                        break;
                }

                mutatedChilds.Add(new ChromosomeInformation(newEpochs, 
                                                            newBatchSize, 
                                                            newLearningRate, 
                                                            GetFitnessValue((newEpochs, newBatchSize, newLearningRate))));
            }

            Console.WriteLine(" Done mutating & calculating fitness");

            return mutatedChilds;
        }

        private int GetLearningRateIntervalNumber(float learningRate)
        {
            var lr = learningRate;
            int counts = 0;
            while(lr < 1)
            {
                lr *= 10;
                counts++;
            }
            return counts % _learningRateMax;
        }

        private (int epochs, int batchSize, float learningRate) GenerateRandomChromosome()
        {
            var rand = new Random();

            int epochs = rand.Next(_epochMin, _epochMax);
            int batchSize = rand.Next(_batchSizeMin, _batchSizeMax);
            int lr = rand.Next(_learningRateMin, _learningRateMax);

            return (epochs, batchSize, learningRate: (float)Math.Pow(10, 0 - lr));
        }


        private float GetFitnessValue((int epochs, int batchSize, float learningRate) chromosomeData)
        {
            DeepLearning dl = new DeepLearning(_trainPath, _testPath);
            dl.TrainModel(_trainImages,
                         (msg) => Console.WriteLine(msg),
                         modelArch: Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.ResnetV250,
                         randomizeSeed: _seed, 
                         epochs: chromosomeData.epochs,
                         batchSize: chromosomeData.batchSize,
                         learningRate: chromosomeData.learningRate);

            var results = dl.TestModel(_testImages);
            var fitness = Evaluation.GetNaiveAccuracy(results);

            return fitness;
        }

    }
}
