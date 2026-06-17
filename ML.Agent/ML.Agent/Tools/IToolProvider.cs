using Microsoft.Extensions.AI;

namespace ML.Agent.Tools
{
    public interface IToolProvider
    {
        public List<AITool> GetTools();
    }
}
