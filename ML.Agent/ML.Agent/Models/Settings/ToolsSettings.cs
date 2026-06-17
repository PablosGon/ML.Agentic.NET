namespace ML.Agent.Models.Settings
{
    public class ToolsSettings
    {
        public HttpToolUrls Urls { get; set; } = new HttpToolUrls();
    }

    public class HttpToolUrls
    {
        public string HousingUrl { get; set; } = string.Empty;

    }
}
