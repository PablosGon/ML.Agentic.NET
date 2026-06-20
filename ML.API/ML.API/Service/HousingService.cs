using AutoMapper;
using ML.API.Data;
using ML.API.Models.Housing;

namespace ML.API.Service
{
    public class HousingService : IInferenceService<HousingRequest, HousingResponse>
    {
        private readonly IModelRunner<HousingInput, HousingOutput> _modelRunner;
        private readonly IMapper _mapper;

        public HousingService(IModelRunner<HousingInput, HousingOutput> modelRunner, IMapper mapper)
        {
            _modelRunner = modelRunner;
            _mapper = mapper;
        }

        public HousingResponse Predict(HousingRequest request)
        {
            var modelInput = _mapper.Map<HousingInput>(request);
            var result = _modelRunner.Predict(modelInput);
            return _mapper.Map<HousingResponse>(result);
        }
    }
}
