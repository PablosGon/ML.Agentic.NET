using ML.API.Domain;

namespace ML.API.Service
{
    public interface IInferenceService<TInput, TOutput>
    {
        TOutput Predict(TInput input);
    }
}
