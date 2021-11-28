using System;
using System.Collections.Generic;
using System.Text;

namespace DetectingAppleDiseases
{
    public class Evaluation
    {
        HashSet<string> ClassTags;
        List<List<double>> finalMatrix;

        public Evaluation() { }

        public static float GetNaiveAccuracy(IEnumerable<ImagePrediction> testData)
        {
            int wordsHit = 0;
            int nrOfWords = 0;
            foreach(var prediction in testData)
            {
                if (prediction.Label == prediction.PredictedLabel)
                    wordsHit++;
                nrOfWords++;
            }

            float accuracy = (float)wordsHit / nrOfWords;
            return accuracy;
        }

        public void CreateSupervizedEvaluationsMatrix(IEnumerable<ImagePrediction> testData, int fbeta = 1)
        {
            ClassTags = new HashSet<string>();
            finalMatrix = new List<List<double>>();

            foreach (var item in testData)
                this.ClassTags.Add(item.Label);

            foreach (var item in testData)
                this.ClassTags.Add(item.PredictedLabel);

            foreach (var tag in this.ClassTags)
            {
                int tp = 0, fp = 0, fn = 0, tn = 0;
                foreach(var curr in testData)
                {
                    if (curr.Label != tag && curr.PredictedLabel != tag)
                        tn++;
                    else if (curr.Label == tag && curr.PredictedLabel == tag)
                        tp++;
                    else if (curr.Label == tag && curr.PredictedLabel != tag)
                        fn++;
                    else if (curr.Label != tag && curr.PredictedLabel == tag)
                        fp++;
                }
                float accuracy = (float)(tp + tn) / (tp + tn + fn + fp);
                if (float.IsNaN(accuracy) || float.IsInfinity(accuracy))
                    accuracy = 0.0f;
                float precision = (float)tp / (tp + fp);
                if (float.IsNaN(precision) || float.IsInfinity(precision))
                    precision = 0.0f;
                float recall = (float)tp / (tp + fn); // true positive rate
                if (float.IsNaN(recall) || float.IsInfinity(recall))
                    recall = 0.0f;
                float fmeasure = (float)((fbeta * fbeta + 1) * precision * recall) / ((fbeta * fbeta) * precision + recall);
                if (float.IsNaN(fmeasure) || float.IsInfinity(fmeasure))
                    fmeasure = 0.0f;
                float specificity = (float)tn / (tn + fp); // true negative rate
                if (float.IsNaN(specificity) || float.IsInfinity(specificity))
                    specificity = 0.0f;
                finalMatrix.Add(new List<double>() { Math.Round(accuracy * 100, 3) , Math.Round(precision * 100, 3) , Math.Round(recall * 100, 3),
                    Math.Round(specificity * 100, 3), Math.Round(fmeasure * 100, 3)  });
            }
        }

        public HashSet<string> GetClassTags()
        {
            return this.ClassTags;
        }

        public List<List<double>> GetClassificationMatrix()
        {
            return this.finalMatrix;
        }

        public int GetFullMatrixLineLength()
        {
            return this.ClassTags.Count + 1;
        }

        public int GetFullMatrixColLength()
        {
            // tag, acc, prec, recall, f1-score, specificity
            return 6;
        }

        public List<List<string>> PrintClassificationResultsMatrix()
        {
            List<List<string>> matrix = new List<List<string>>();
            int i = 0;
            foreach (string hashTag in this.ClassTags)
            {
                matrix.Add(new List<string>()
                {
                    hashTag,
                    this.finalMatrix[i][0].ToString(),
                    this.finalMatrix[i][1].ToString(),
                    this.finalMatrix[i][2].ToString(),
                    this.finalMatrix[i][3].ToString(),
                    this.finalMatrix[i][4].ToString()
                });
                i++;
            }

            double totalAccuracy = 0.0f, totalPrecision = 0.0f, totalRecall = 0.0f, totalFmeasure = 0.0f, totalSpecificity = 0.0f;
            for (int j = 0; j < ClassTags.Count; j++)
            {
                totalAccuracy += finalMatrix[j][0];
                totalPrecision += finalMatrix[j][1];
                totalRecall += finalMatrix[j][2];
                totalSpecificity += finalMatrix[j][3];
                totalFmeasure += finalMatrix[j][4];
            }
            totalAccuracy = (double)(totalAccuracy / ClassTags.Count);
            totalAccuracy = Math.Round(totalAccuracy, 3);
            totalPrecision = (double)(totalPrecision / ClassTags.Count);
            totalPrecision = Math.Round(totalPrecision, 3);
            totalRecall = (double)(totalRecall / ClassTags.Count);
            totalRecall = Math.Round(totalRecall, 3);
            totalSpecificity = (double)(totalSpecificity / ClassTags.Count);
            totalSpecificity = Math.Round(totalSpecificity, 3);
            totalFmeasure = (double)(totalFmeasure / ClassTags.Count);
            totalFmeasure = Math.Round(totalFmeasure, 3);


            matrix.Add(new List<string>()
            {
                "TOTAL",
                totalAccuracy.ToString(),
                totalPrecision.ToString(),
                totalRecall.ToString(),
                totalSpecificity.ToString(),
                totalFmeasure.ToString()
            });

            return matrix;
        }
    }
}
