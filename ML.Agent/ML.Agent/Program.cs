using A2A.AspNetCore;
using Amazon;
using Amazon.BedrockRuntime;
using Amazon.Runtime;
using Amazon.Runtime.CredentialManagement;
using Microsoft.Agents.AI;
using Microsoft.Extensions.AI;
using ML.Agent.Agent;
using ML.Agent.Models.Settings;
using ML.Agent.Services;
using ML.Agent.Services.Interfaces;
using ML.Agent.Tools;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddControllers();
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

builder.Services.Configure<AgentSettings>(builder.Configuration.GetSection("Agent"));
builder.Services.Configure<BedrockSettings>(builder.Configuration.GetSection("Bedrock"));
builder.Services.Configure<ToolsSettings>(builder.Configuration.GetSection("Tools"));

builder.Services.AddScoped<IChatService, ChatService>();
builder.Services.AddSingleton<IToolProvider, ToolProvider>();
builder.Services.AddSingleton<IAgentFactory, AgentFactory>();
builder.Services.AddHttpClient();

builder.Services.AddKeyedSingleton<AIAgent>("agent", (sp, _) =>
{
    var agentFactory = sp.GetRequiredService<IAgentFactory>();
    return agentFactory.Create();
});

builder.AddA2AServer(builder.Configuration.GetValue<string>("Agent:Name"));

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
