using Microsoft.AspNetCore.Mvc;
using ML.API.Domain;
using ML.API.Service;

namespace ML.API.Controllers
{
    [ApiController]
    public class HousingController : ControllerBase
    {
        private readonly IInferenceService<HousePricing, double> _inferenceService;

        public HousingController(IInferenceService<HousePricing, double> inferenceService)
        {
            _inferenceService = inferenceService;
        }

        [HttpPost]
        [Route("/housing")]
        public ActionResult<double> HousingInference([FromBody] HousePricing request)
        {
            return _inferenceService.Predict(request);
        }
    }
}
