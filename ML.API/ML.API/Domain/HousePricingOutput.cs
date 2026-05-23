using Microsoft.ML.Data;
using System.Numerics;

namespace ML.API.Domain
{
    public class HousePricingOutput
    {
        [ColumnName("Score")]
        public float HousePrice { get; set; }
    }
}
