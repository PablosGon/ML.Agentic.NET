using CsvHelper.Configuration.Attributes;
using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.NaiveBayes.Domain
{
    internal class Email
    {
        [Name("Email Text")]
        public string Text { get; set; } = string.Empty;

        [Name("Email Type")]
        public string Type { get; set; } = string.Empty;
    }
}
