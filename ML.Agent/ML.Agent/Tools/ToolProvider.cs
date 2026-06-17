using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using ML.Agent.Models.Prediction;
using ML.Agent.Models.Settings;

namespace ML.Agent.Tools
{
    public class ToolProvider : IToolProvider
    {
        private HttpClient _httpClient;
        private HttpToolUrls _urls;

        public ToolProvider(HttpClient httpClient, IOptions<ToolsSettings> settings)
        {
            _httpClient = httpClient;
            _urls = settings.Value.Urls;
        }

        public List<AITool> GetTools()
        {
            return [AIFunctionFactory.Create(GetHousePrediction)];
        }

        private async Task<double> GetHousePrediction(Housing housing)
        {
            var response = await _httpClient.PostAsJsonAsync(_urls.HousingUrl, housing);
            var stringResponse = await response.Content.ReadAsStringAsync();
            return Convert.ToDouble(stringResponse);
        }
    }
}
