namespace ML.Agent.Models.Settings
{
    public class ToolsSettings
    {
        public HttpToolUrls Urls = new HttpToolUrls();
    }

    public class HttpToolUrls
    {
        public string HousingUrl { get; set; } = string.Empty;

    }
}
