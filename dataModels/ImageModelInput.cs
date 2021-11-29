using System;

namespace DetectingAppleDiseases
{
    public class ImageModelInput
    {
        public byte[] Image { get; set; }

        public uint LabelAsKey { get; set; }

        public string Name { get; set; }

        public string Label { get; set; }
    }
}