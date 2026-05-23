namespace ML.API.Data
{
    public interface IModelRunner<TInput, TOutput>
    {
        public TOutput Predict(TInput input);
    }
}
