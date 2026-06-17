using Amazon;
using Amazon.BedrockRuntime;
using Amazon.Runtime;
using Amazon.Runtime.CredentialManagement;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using Microsoft.Extensions.Options;
using ML.Agent.Models.Settings;
using ML.Agent.Tools;

namespace ML.Agent.Agent
{
    public class AgentFactory : IAgentFactory
    {
        private IToolProvider _toolProvider;
        private AgentSettings _agentSettings;
        private BedrockSettings _bedrockSettings;

        public AgentFactory(IToolProvider toolProvider, IOptions<AgentSettings> agentSettings, IOptions<BedrockSettings> bedrockSettings)
        {
            _toolProvider = toolProvider;
            _agentSettings = agentSettings.Value;
            _bedrockSettings = bedrockSettings.Value;
        }

        public AIAgent Create()
        {
            var region = RegionEndpoint.GetBySystemName(_bedrockSettings.RegionName);
            var awsCredentials = GetAwsCredentials(_bedrockSettings.ProfileName);
            var bedrockClient = new AmazonBedrockRuntimeClient(awsCredentials, region);
            var chatClient = bedrockClient.AsIChatClient(_bedrockSettings.ModelId);
            var tools = _toolProvider.GetTools();

            return chatClient.AsAIAgent(
                name: _agentSettings.Name,
                description: _agentSettings.Description,
                tools: tools);
        }

        private static AWSCredentials GetAwsCredentials(string profile)
        {
            ArgumentNullException.ThrowIfNullOrEmpty(profile, nameof(profile));

            var chain = new CredentialProfileStoreChain();
            if (!chain.TryGetAWSCredentials(profile, out var credentials))
            {
                throw new Exception($"Could not get credentials for profile {profile}");
            }

            return credentials;
        }
    }
}
