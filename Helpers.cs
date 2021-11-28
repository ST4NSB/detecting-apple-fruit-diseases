using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;

namespace DetectingAppleDiseases
{
    public static class Helpers
    {
        private static string[] GetDatasetImages(string evaluationType)
        {
            var folder = Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "dataset", evaluationType);
            var files = Directory.GetFiles(folder, "*", SearchOption.AllDirectories);
            return files;
        }

        public static IEnumerable<ImageData> GetTrainingImages()
        {
            var files = GetDatasetImages(evaluationType: "Train");
            return files.Select(file => new ImageData
            {
                Name = file,
                Label = Directory.GetParent(file).Name
            });
        }

        public static IEnumerable<ImageModelInput> GetTestImages()
        {
            var files = GetDatasetImages(evaluationType: "Test");
            return files.Select(file => new ImageModelInput
            {
                Name = file,
                Label = Directory.GetParent(file).Name
            });
        }

        public static string GetDatasetImagesPath(string evaluationType)
        {
            return Path.Combine(Environment.CurrentDirectory, "..", "..", "..", "dataset", evaluationType);
        }
    }
}
