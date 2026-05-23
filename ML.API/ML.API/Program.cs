using Microsoft.Extensions.ML;
using ML.API.Data;
using ML.API.Domain;
using ML.API.Service;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddScoped<IInferenceService<HousePricing, double>, HousingService>();

builder.Services.AddScoped<IModelRunner<HousePricing, HousePricingOutput>, HousingModelRunner>();

builder.Services.AddPredictionEnginePool<HousePricing, HousePricingOutput>()
    .FromFile("HousePricingModel.zip");

builder.Services.AddControllers();
// Learn more about configuring OpenAPI at https://aka.ms/aspnet/openapi
builder.Services.AddOpenApi();

var app = builder.Build();

// Configure the HTTP request pipeline.
if (app.Environment.IsDevelopment())
{
    app.MapOpenApi();
}

app.UseHttpsRedirection();

app.UseAuthorization();

app.MapControllers();

app.Run();
