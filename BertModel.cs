using Tensorflow;

namespace AI_WF_MARKETING
{
    public class BertModel
    {
        private readonly Session _session;
        private readonly Graph _graph;

        public BertModel(string modelDir)
        {
            // e.g., "models/my_bert_savedmodel"
            var loader = new SavedModelLoader(modelDir);
            _session = loader.Load();
            _graph = _session.graph;
        }

        public Tensor RunInference(Tensor inputIds, Tensor attentionMask, Tensor tokenTypeIds)
        {
            // The node names might differ. Inspect with saved_model_cli or Python to confirm.
            // For official BERT, you might see "serving_default_input_ids" or "input_1", etc.
            // We'll guess "input_ids", "attention_mask", "token_type_ids" for now.
            var runner = _session.GetRunner();
            runner
                .AddInput(_graph["input_ids"][0], inputIds)
                .AddInput(_graph["attention_mask"][0], attentionMask)
                .AddInput(_graph["token_type_ids"][0], tokenTypeIds)
                .Fetch(_graph["output"][0]); // or "logits" / "sequence_output" / "pooled_output"

            var outputTensors = runner.Run();
            return outputTensors[0];
        }
    }
}
