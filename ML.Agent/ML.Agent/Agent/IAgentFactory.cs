using Microsoft.Agents.AI;

namespace ML.Agent.Agent
{
    public interface IAgentFactory
    {
        public AIAgent Create();
    }
}
