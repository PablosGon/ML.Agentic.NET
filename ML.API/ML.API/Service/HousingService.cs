using Microsoft.ML;
using ML.API.Data;
using ML.API.Domain;

namespace ML.API.Service
{
    public class HousingService : IInferenceService<HousePricing, double>
    {
        private readonly IModelRunner<HousePricing, HousePricingOutput> _modelRunner;

        public HousingService(IModelRunner<HousePricing, HousePricingOutput> modelRunner)
        {
            _modelRunner = modelRunner;
        }

        public double Predict(HousePricing housePricing)
        {
            var result = _modelRunner.Predict(housePricing);
            return result.HousePrice;
        }
    }
}
