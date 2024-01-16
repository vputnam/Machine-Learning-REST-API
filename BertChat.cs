using System;
using System.IO;
using System.Threading.Tasks;
using System.Collections.Generic;
using Microsoft.AspNetCore.Mvc;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Extensions.Http;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Newtonsoft.Json;

namespace MachineLearningFunctions
{
    public static class BertChat 
    {
        [FunctionName("BertChat")]
        public static async Task<IActionResult> Run(
            [HttpTrigger(AuthorizationLevel.Anonymous, "get", "post", Route = null)] HttpRequest req,
            ILogger log)
        {
            log.LogInformation("C# HTTP trigger function processed a request.");

            string name = req.Query["name"];

            string requestBody = await new StreamReader(req.Body).ReadToEndAsync();
            dynamic data = JsonConvert.DeserializeObject(requestBody);
            name = name ?? data?.name;

            // Get path to model to create inference session.
            var modelPath = "bert_model.onnx";

            BertProcessor bertProcessor = new BertProcessor();
            BertInput tokenizer = bertProcessor.BertTokenize(data);

            var runOptions = new RunOptions();
            var session = new InferenceSession(modelPath);

            Dictionary<string, OrtValue> inputs = bertProcessor.BertCreateInputs(tokenizer, runOptions, session, modelPath);

            var output = session.Run(runOptions, inputs, session.OutputNames);

            string responseMessage = string.IsNullOrEmpty(name)
                ? "This HTTP triggered function executed successfully. Soon to be calling Bert model."
                : $"Hello, {name}. This HTTP triggered function executed successfully.";

            return new OkObjectResult(responseMessage);
        }
    }
}
