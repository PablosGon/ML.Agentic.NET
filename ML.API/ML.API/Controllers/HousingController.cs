using Microsoft.AspNetCore.Mvc;
using ML.API.Models.Housing;
using ML.API.Service;

namespace ML.API.Controllers
{
    [ApiController]
    public class HousingController : ControllerBase
    {
        private readonly IInferenceService<HousingRequest, HousingResponse> _inferenceService;

        public HousingController(IInferenceService<HousingRequest, HousingResponse> inferenceService)
        {
            _inferenceService = inferenceService;
        }

        [HttpPost]
        [Route("/housing")]
        public ActionResult<HousingResponse> HousingInference([FromBody] HousingRequest request)
        {
            return _inferenceService.Predict(request);
        }
    }
}
