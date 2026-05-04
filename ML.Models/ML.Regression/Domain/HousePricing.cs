using Microsoft.ML.Data;

namespace ML.Regression.Domain
{
    public class HousePricing
    {
        [LoadColumn(0)]
        public float SquareFootage { get; set; }

        [LoadColumn(1)]
        public float NumBedrooms { get; set; }

        [LoadColumn(2)]
        public float NumBathrooms { get; set; }

        [LoadColumn(3)]
        public float YearBuilt { get; set; }

        [LoadColumn(4)]
        public float LotSize { get; set; }

        [LoadColumn(5)]
        public float GarageSize { get; set; }

        [LoadColumn(6)]
        public string NeighbourhoodQuality { get; set; } = string.Empty;

        [LoadColumn(7)]
        public float HousePrice { get; set; }
    }
}
