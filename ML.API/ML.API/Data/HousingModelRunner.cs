using Microsoft.Extensions.ML;
using ML.API.Models.Housing;

namespace ML.API.Data
{
    public class HousingModelRunner : IModelRunner<HousingInput, HousingOutput>
    {
        private readonly PredictionEnginePool<HousingInput, HousingOutput> _housingEnginePool;

        public HousingModelRunner(PredictionEnginePool<HousingInput, HousingOutput> housingEnginePool)
        {
            _housingEnginePool = housingEnginePool;
        }

        public HousingOutput Predict(HousingInput input)
        {
            return _housingEnginePool.Predict(input);
        }
    }
}
