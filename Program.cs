using System;
using System.Linq;

namespace DetectingAppleDiseases
{
    class Program
    {
        static void Main(string[] args)
        {
            GeneticOptimizer go = new GeneticOptimizer();
            go.StartGeneticOptimizationProcess();
        }
    }
}
