using Microsoft.Extensions.ML;
using ML.API.Domain;

namespace ML.API.Data
{
    public class HousingModelRunner : IModelRunner<HousePricing, HousePricingOutput>
    {
        private readonly PredictionEnginePool<HousePricing, HousePricingOutput> _housingEnginePool;

        public HousingModelRunner(PredictionEnginePool<HousePricing, HousePricingOutput> housingEnginePool)
        {
            _housingEnginePool = housingEnginePool;
        }

        public HousePricingOutput Predict(HousePricing input)
        {
            return _housingEnginePool.Predict(input);
        }
    }
}
