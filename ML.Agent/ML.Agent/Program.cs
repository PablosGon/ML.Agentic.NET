using A2A.AspNetCore;
using Amazon;
using Amazon.BedrockRuntime;
using Amazon.Runtime;
using Amazon.Runtime.CredentialManagement;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ML.Agent.Services;
using ML.Agent.Services.Interfaces;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

builder.Services.AddScoped<IChatService, ChatService>();

builder.Services.AddKeyedSingleton<AIAgent>("agent", (sp, _) =>
{
    var region = RegionEndpoint.USEast1;
    var awsCredentials = GetAwsCredentials("default");
    var bedrockClient = new AmazonBedrockRuntimeClient(awsCredentials, region);
    var chatClient = bedrockClient.AsIChatClient("modelID");

    var tools = new List<AITool>();

    return chatClient.AsAIAgent(
        instructions: "You are an agent",
        name: "agent",
        description: "",
        tools: tools);
});

builder.AddA2AServer("agent");

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.MapA2AHttpJson("agent", "/a2a/agent");

app.MapWellKnownAgentCard(new A2A.AgentCard
{

});

app.Run();

static AWSCredentials GetAwsCredentials(string profile)
{
    ArgumentNullException.ThrowIfNullOrEmpty(profile, nameof(profile));

    var chain = new CredentialProfileStoreChain();
    if (!chain.TryGetAWSCredentials(profile, out var credentials))
    {
        throw new Exception($"Could not get credentials for profile {profile}");
    }

    return credentials;
}