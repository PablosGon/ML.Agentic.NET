using Microsoft.AspNetCore.Mvc;
using ML.Agent.Models;
using ML.Agent.Services.Interfaces;

namespace ML.Agent.Controllers
{
    [ApiController]
    public class ChatController(IChatService service) : ControllerBase
    {
        [HttpPost]
        public async Task<ActionResult<ChatResponse>> Chat(ChatRequest request)
        {
            try
            {
                var result = await service.Chat(request);
                return Ok(result);
            }
            catch (ArgumentException ex)
            {
                return BadRequest(ex.Message);
            }
            catch(Exception ex)
            {
                return StatusCode(500, "Internal server error");
            }
        }
    }
}
