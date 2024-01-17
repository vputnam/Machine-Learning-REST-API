
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

        private BertUncasedLargeTokenizer tokenizer = new BertUncasedLargeTokenizer();
        private List<(string Token, int VocabularyIndex, long SegmentIndex)> tokens;
        private RunOptions runOptions = new RunOptions();
        private InferenceSession session;
        public BertProcessor(String modelPath)
        {
            this.session = new InferenceSession(modelPath);
        }

        public BertInput BertTokenize(string sentence)
        {
            // Get the sentence tokens.
            tokens = tokenizer.Tokenize(sentence);

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

        public String BertPostProcessor(BertInput bertInput)
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

            var output = session.Run(runOptions, inputs, session.OutputNames);

            var startLogits = output[0].GetTensorDataAsSpan<float>();
            int startIndex = GetMaxValueIndex(startLogits);

            var endLogits = output[output.Count - 1].GetTensorDataAsSpan<float>();
            int endIndex = GetMaxValueIndex(endLogits);

            var predictedTokens = tokens
                          .Skip(startIndex)
                          .Take(endIndex + 1 - startIndex)
                          .Select(o => tokenizer.IdToToken((int)o.VocabularyIndex))
                          .ToList();

            // Print the result.
            Console.WriteLine(String.Join(" ", predictedTokens));
            return String.Join(" ", predictedTokens);
        }

        private int GetMaxValueIndex(ReadOnlySpan<float> span)
        {
            float maxVal = span[0];
            int maxIndex = 0;
            for (int i = 1; i < span.Length; ++i)
            {
                var v = span[i];
                if (v > maxVal)
                {
                    maxVal = v;
                    maxIndex = i;
                }
            }
            return maxIndex;
        }

    }

}