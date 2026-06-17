using Amazon;
using Amazon.BedrockRuntime;
using Amazon.Runtime;
using Amazon.Runtime.CredentialManagement;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ML.Agent.Tools;

namespace ML.Agent.Agent
{
    public class AgentFactory(IToolProvider toolProvider) : IAgentFactory
    {
        public AIAgent Create()
        {
            var region = RegionEndpoint.USEast1;
            var awsCredentials = GetAwsCredentials("default");
            var bedrockClient = new AmazonBedrockRuntimeClient(awsCredentials, region);
            var chatClient = bedrockClient.AsIChatClient("modelID");
            var tools = toolProvider.GetTools();

            return chatClient.AsAIAgent(
                instructions: "You are an agent",
                name: "agent",
                description: "",
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
