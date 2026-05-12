using CsvHelper.Configuration.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.LSTM.TensorFlow.Domain
{
    internal class ImdbReview
    {
        [Name("review")]
        public string Review { get; set; } = string.Empty;

        [Name("sentiment")]
        public string Sentiment { get; set; } = string.Empty;
    }
}
