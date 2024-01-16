
using BERTTokenizers;
using Microsoft.ML.Data;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using System.Collections.Generic;
using System.Linq;
using System;

namespace MachineLearningFunctions
{
    public class BertProcessor
    {
        public BertProcessor()
        {

        }

        public BertInput BertTokenize(string input){

            var sentence = "{\"question\": \"Where is Bob Dylan From?\", \"context\": \"Bob Dylan is from Duluth, Minnesota and is an American singer-songwriter\"}";

            // Create Tokenizer and tokenize the sentence.
            var tokenizer = new BertUncasedLargeTokenizer();

            // Get the sentence tokens.
            var tokens = tokenizer.Tokenize(sentence);
            // Console.WriteLine(String.Join(", ", tokens));

            // Encode the sentence and pass in the count of the tokens in the sentence.
            var encoded = tokenizer.Encode(tokens.Count(), sentence);

            // Break out encoding to InputIds, AttentionMask and TypeIds from list of (input_id, attention_mask, type_id).
            var bertInput = new BertInput()
            {
                InputIds = encoded.Select(t => t.InputIds).ToArray(),
                AttentionMask = encoded.Select(t => t.AttentionMask).ToArray(),
                TypeIds = encoded.Select(t => t.TokenTypeIds).ToArray(),
            };
            return bertInput;
        }

        public Dictionary<string, OrtValue> BertCreateInputs(BertInput bertInput,
            RunOptions runOptions, 
            InferenceSession session,
            string modelPath)
        {

            // Create input tensors over the input data.
            using var inputIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.InputIds,
                    new long[] { 1, bertInput.InputIds.Length });

            using var attMaskOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.AttentionMask,
                    new long[] { 1, bertInput.AttentionMask.Length });

            using var typeIdsOrtValue = OrtValue.CreateTensorValueFromMemory(bertInput.TypeIds,
                    new long[] { 1, bertInput.TypeIds.Length });

            // Create input data for session. Request all outputs in this case.
            var inputs = new Dictionary<string, OrtValue>
            {
                { "input_ids", inputIdsOrtValue },
                { "input_mask", attMaskOrtValue },
                { "segment_ids", typeIdsOrtValue }
            };

            return inputs;
        }

    }

}