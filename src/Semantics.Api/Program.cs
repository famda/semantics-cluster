using Azure.Storage.Blobs;
using Microsoft.AspNetCore.Http.Features;
using System.Text.RegularExpressions;

const string Container = "files";

var builder = WebApplication.CreateBuilder(args);

builder.Services.AddOpenApi();
builder.WebHost.ConfigureKestrel(options => options.Limits.MaxRequestBodySize = 1024L * 1024L * 1024L); //1GB
builder.Services.Configure<FormOptions>(options => options.MultipartBodyLengthLimit = 1024L * 1024L * 1024L);

var azuriteConnStr = builder.Configuration["AZURITE_CONN_STR"] ?? throw new InvalidOperationException("Azurite connection string not found in configuration or environment variables.");
var rayBaseUrl = builder.Configuration["RAY_DASHBOARD_URL"] ?? throw new InvalidOperationException("Ray Dashboard URL not found in configuration or environment variables.");

builder.Services.AddSingleton(new BlobServiceClient(azuriteConnStr));
builder.Services.AddHttpClient("ray", c => c.BaseAddress = new Uri(rayBaseUrl));

var app = builder.Build();

var blobService = app.Services.GetRequiredService<BlobServiceClient>();
var container = blobService.GetBlobContainerClient(Container);
await container.CreateIfNotExistsAsync();

if (app.Environment.IsDevelopment()) {
    app.MapOpenApi();
    app.UseSwaggerUI(options => {
        options.SwaggerEndpoint("/openapi/v1.json", "v1");
    });
}

app.MapGet("/", () => Results.Redirect("/swagger")).ExcludeFromDescription();

app.MapPost("/transcribe", async (IFormFile file, IHttpClientFactory httpClientFactory, BlobServiceClient blobs) => {

    if (file.Length == 0)
        return Results.BadRequest(new { error = "No file uploaded." });

    var jobId = Guid.NewGuid().ToString("N");
    var blobName = Regex.Replace(file.FileName, @"[^\w\s.\-]", "_");

    // 1. Upload file to Azurite
    var containerClient = blobs.GetBlobContainerClient(Container);
    await containerClient.CreateIfNotExistsAsync();

    var blobClient = containerClient.GetBlobClient($"{jobId}/{blobName}");

    await using var stream = file.OpenReadStream();
    await blobClient.UploadAsync(stream, overwrite: true);

    // 2. Submit Ray Job
    var rayClient = httpClientFactory.CreateClient("ray");

    var jobPayload = new {
        entrypoint = $"python /opt/jobs/transcribe.py --job-id {jobId} --blob-name \"{blobName}\"",
        submission_id = $"transcription-{jobId}",
        //runtime_env = new {
        //    env_vars = new Dictionary<string, string> {
        //        ["WHISPER_MODEL"] = whisperModel,
        //    }
        //},
        entrypoint_num_cpus = 0,
        entrypoint_num_gpus = 0,
        metadata = new Dictionary<string, string> {
            ["job_id"] = jobId,
            ["source_file"] = blobName,
        }
    };

    var response = await rayClient.PostAsJsonAsync("/api/jobs/", jobPayload);
    var body = await response.Content.ReadAsStringAsync();

    if (!response.IsSuccessStatusCode)
        return Results.Problem($"Ray job submission failed: {body}", statusCode: 502);

    return Results.Accepted(value: new {
        jobId,
        raySubmissionId = $"transcription-{jobId}",
        status = "PENDING",
        blobPath = $"{Container}/{jobId}/{blobName}",
        resultPath = $"{Container}/{jobId}/transcription.json",
    });
})
.DisableAntiforgery();

app.Run();
