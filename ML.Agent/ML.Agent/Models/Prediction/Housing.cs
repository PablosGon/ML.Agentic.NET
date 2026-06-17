namespace ML.Agent.Models.Prediction
{
    public class Housing
    {
        public float SquareFootage { get; set; }
        public float NumBedrooms { get; set; }
        public float NumBathrooms { get; set; }
        public float YearBuilt { get; set; }
        public float LotSize { get; set; }
        public float GarageSize { get; set; }
        public string NeighborhoodQuality { get; set; } = string.Empty;
        public float HousePrice { get; set; }
    }
}
