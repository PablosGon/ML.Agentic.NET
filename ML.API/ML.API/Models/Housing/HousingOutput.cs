using Microsoft.ML.Data;

namespace ML.API.Models.Housing
{
    public class HousingOutput
    {
        [ColumnName("Score")]
        public float HousePrice { get; set; }
    }
}
