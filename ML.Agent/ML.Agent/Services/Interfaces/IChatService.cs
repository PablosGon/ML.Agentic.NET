using ML.Agent.Models;

namespace ML.Agent.Services.Interfaces
{
    public interface IChatService
    {
        Task<ChatResponse> Chat(ChatRequest request); 
    }
}
