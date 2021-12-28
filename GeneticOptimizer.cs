using System;
using System.Collections.Generic;
using System.Linq;

namespace DetectingAppleDiseases
{
    public class GeneticOptimizer
    {
        private const int _seed = 42;
        private const int _epochMin = 1, _epochMax = 128;
        private const int _batchSizeMin = 1, _batchSizeMax = 128; 
        private const int _learningRateMin = 1, _learningRateMax = 8;

        private const float _fitnessThreshold = 8.25f, _mutationThreshold = 0.4f;

        private DeepLearning _deepLearningModel;
        private int _initialPopulation, _populationSelection, _generationTotalNumbers;
        private IEnumerable<ImageData> _trainImages;
        private IEnumerable<ImageModelInput> _testImages;
        private string _testPath, _trainPath;

        private struct ChromosomeInformation
        {
            public int Epochs;
            public int BatchSize;
            public float LearningRate;
            public float Fitness;

            public ChromosomeInformation(int epochs, int batchSize, float learningRate, float fitness = float.MinValue)
            {
                Epochs = epochs;
                BatchSize = batchSize;
                LearningRate = learningRate;
                Fitness = fitness;
            }
        }

        public GeneticOptimizer(int initialPopulation = 40, int populationSelection = 20, int totalNumberOfGenerations = 15)
        {
            _deepLearningModel = new DeepLearning(_trainPath, _testPath);

            _trainPath = Helpers.GetDatasetImagesPath(evaluationType: "Train");
            _trainImages = Helpers.GetTrainingImages();

            _testPath = Helpers.GetDatasetImagesPath(evaluationType: "Test");
            _testImages = Helpers.GetTestImages();

            _initialPopulation = initialPopulation;
            _populationSelection = populationSelection;
            _generationTotalNumbers = totalNumberOfGenerations;
        }

        public void StartGeneticOptimizationProcess()
        {
            Console.WriteLine(" *** Wait a few minutes .. This make take a while..");
            var chromosomes = new List<ChromosomeInformation>();
            for (int i = 0; i < _initialPopulation; i++)
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
            chromosomes = chromosomes.Take(_populationSelection).ToList();

            var generation = 1;
            while (generation <= _generationTotalNumbers)
            {
                Console.WriteLine(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                Console.WriteLine($" Generation number: {generation++}");
                PrintInformations(chromosomes.First());

                var childs = ApplyCrossover(chromosomes);
                var mutatedChilds = MutateChilds(childs);
                Console.WriteLine(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
                chromosomes = mutatedChilds.OrderByDescending(x => x.Fitness).ToList();
                chromosomes = chromosomes.Take(_populationSelection).ToList();

                if (chromosomes.First().Fitness >= _fitnessThreshold) break;
            }

            PrintInformations(chromosomes.First());
        }

        private void PrintInformations(ChromosomeInformation chromosome)
        {
            Console.WriteLine(" *** Chromosome configuration ***");
            Console.WriteLine($" Number of epochs: {chromosome.Epochs}");
            Console.WriteLine($" Batch size: {chromosome.BatchSize}");
            Console.WriteLine($" Learning rate: {chromosome.LearningRate}");
            Console.WriteLine($" ! Fitness (Accuracy): {chromosome.Fitness}");
            Console.WriteLine(" *** ------------------------ ***");
        }

        private List<ChromosomeInformation> ApplyCrossover(List<ChromosomeInformation> parents)
        {
            var rand = new Random();
            var childs = new List<ChromosomeInformation>();

            for (int i = 0; i < parents.Count; i += 2)
            {
                var (firstChildEpoch, secondChildEpoch) = GetCrossOverOffspring(parents[i].Epochs, parents[i+1].Epochs, bitsToPass: 1, rand.Next(7));
                var (firstChildBatchSize, secondChildBatchSize) = GetCrossOverOffspring(parents[i].BatchSize, parents[i+1].BatchSize, bitsToPass: 1, rand.Next(7));
                var (firstChildLearningRate, secondChildLearningRate) = GetCrossOverOffspring(GetLearningRateIntervalNumber(parents[i].LearningRate),
                                                                                              GetLearningRateIntervalNumber(parents[i+1].LearningRate), 
                                                                                              bitsToPass: 5, 
                                                                                              rand.Next(3));
                
                childs.Add(new ChromosomeInformation(firstChildEpoch, firstChildBatchSize, (float)Math.Pow(10, 0 - firstChildLearningRate)));
                childs.Add(new ChromosomeInformation(secondChildEpoch, secondChildBatchSize, (float)Math.Pow(10, 0 - secondChildLearningRate)));
            }

            Console.WriteLine($" Crossover completed! Number of children: {childs.Count}");

            return childs;
        }

        // one point crossover
        private (int firstChild, int secondChild) GetCrossOverOffspring(int parent1, int parent2, int bitsToPass, int randValue)
        {
            byte p1Byte = (byte)parent1;
            byte p2Byte = (byte)parent2;
            int res1 = 0, res2 = 0;
            int passedBits = 0, randCounting = 0;

            while (passedBits < 8)
            {
                if (passedBits < bitsToPass)
                {
                    passedBits++;
                    p1Byte <<= 1;
                    p2Byte <<= 1;
                    continue;
                }

                var msbP1 = (p1Byte & 0x80) != 0;
                var msbP2 = (p2Byte & 0x80) != 0;

                if (randCounting <= randValue)
                {
                    res1 += (msbP1) ? 1 : 0;
                    res2 += (msbP2) ? 1 : 0;
                }
                else
                {
                    res1 += (msbP2) ? 1 : 0;
                    res2 += (msbP1) ? 1 : 0;
                }

                res1 <<= 1;
                res2 <<= 1;

                p1Byte <<= 1;
                p2Byte <<= 1;

                randCounting++;
                passedBits++;
            }

            res1 >>= 1;
            res2 >>= 1;

            return (res1, res2);
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
               
                var mutationProbability = rand.Next(10) + 1;
                if (mutationProbability <= _mutationThreshold)
                {
                    newEpochs = NegateBit(newEpochs, rand.Next(7));
                    newEpochs = newEpochs < _epochMin ? _epochMin : newEpochs;
                    
                    newBatchSize = NegateBit(newBatchSize, rand.Next(7));
                    newBatchSize = newBatchSize < _batchSizeMin ? _batchSizeMin : newBatchSize;

                    var lrMutated = NegateBit(GetLearningRateIntervalNumber(newLearningRate), rand.Next(3));
                    lrMutated = lrMutated < _learningRateMin ? _learningRateMin : lrMutated;

                    newLearningRate = (float)Math.Pow(10, 0 - lrMutated);
                }

                mutatedChilds.Add(new ChromosomeInformation(newEpochs, 
                                                            newBatchSize, 
                                                            newLearningRate, 
                                                            GetFitnessValue((newEpochs, newBatchSize, newLearningRate))));
            }

            Console.WriteLine(" Done mutating & calculating fitness");

            return mutatedChilds;
        }

        private int NegateBit(int chromosome, int position)
        {
            var newVal = chromosome;
            var negBit = (chromosome & (1 << position)) != 0;
            if (negBit)
            {
                newVal &= ~(1 << position);
            }
            else
            {
                newVal |= 1 << position;
            }
            return newVal;
        }

        private int GetLearningRateIntervalNumber(float learningRate)
        {
            decimal lr = (decimal)learningRate;
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
            _deepLearningModel.TrainModel(_trainImages,
                         (msg) => Console.WriteLine(msg),
                         modelArch: Microsoft.ML.Vision.ImageClassificationTrainer.Architecture.MobilenetV2,
                         randomizeSeed: _seed, 
                         epochs: chromosomeData.epochs,
                         batchSize: chromosomeData.batchSize,
                         learningRate: chromosomeData.learningRate);

            var results = _deepLearningModel.TestModel(_testImages);
            var fitness = Evaluation.GetNaiveAccuracy(results);

            return fitness;
        }
    }
}
