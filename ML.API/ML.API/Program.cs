using Microsoft.Extensions.ML;
using ML.API.Data;
using ML.API.Models.Housing;
using ML.API.Models.Mapping;
using ML.API.Service;

var builder = WebApplication.CreateBuilder(args);

// Add services to the container.

builder.Services.AddScoped<IInferenceService<HousingRequest, HousingResponse>, HousingService>();

builder.Services.AddScoped<IModelRunner<HousingInput, HousingOutput>, HousingModelRunner>();

builder.Services.AddPredictionEnginePool<HousingInput, HousingOutput>()
    .FromFile("HousePricingModel.zip");

builder.Services.AddAutoMapper(x =>
{
    x.AddProfile<HousingProfile>();
});

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
