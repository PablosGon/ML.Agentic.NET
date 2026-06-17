using Microsoft.Agents.AI;
using ML.Agent.Models;
using ML.Agent.Services.Interfaces;

namespace ML.Agent.Services
{
    public class ChatService([FromKeyedServices("agent")] AIAgent agent) : IChatService
    {
        public async Task<ChatResponse> Chat(ChatRequest request)
        {
            var message = request.Message;

            if (string.IsNullOrEmpty(message))
            {
                throw new ArgumentException("Message cannot be null or empty");
            }

            var response = await agent.RunAsync(message);

            return new ChatResponse
            {
                Message = response.Text
            };
        }
    }
}
