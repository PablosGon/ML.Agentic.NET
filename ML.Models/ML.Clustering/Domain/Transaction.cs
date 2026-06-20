using CsvHelper.Configuration.Attributes;
using System;
using System.Collections.Generic;
using System.Text;

namespace ML.Clustering.Domain
{
    public class Transaction
    {
        public string Invoice { get; set; } = string.Empty;
        public string StockCode { get; set; } = string.Empty;
        public string Description { get; set; } = string.Empty;
        public int Quantity { get; set; }
        public DateTime InvoiceDate { get; set; }
        public double Price { get; set; }

        [Name("Customer ID")]
        public string CustomerId { get; set; } = string.Empty;
        public string Country { get; set; } = string.Empty;
    }
}
