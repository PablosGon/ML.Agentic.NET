using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using ML.Agent.Models.Prediction;
using ML.Agent.Models.Settings;

namespace ML.Agent.Tools
{
    public class ToolProvider : IToolProvider
    {
        private HttpClient _httpClient;

        public ToolProvider(HttpClient httpClient)
        {
            _httpClient = httpClient;
        }

        public List<AITool> GetTools()
        {
            return [AIFunctionFactory.Create(GetHousePrediction)];
        }

        private async Task<double> GetHousePrediction(Housing housing)
        {
            var response = await _httpClient.PostAsJsonAsync("url", housing);
            var stringResponse = await response.Content.ReadAsStringAsync();
            return Convert.ToDouble(stringResponse);
        }
    }
}
