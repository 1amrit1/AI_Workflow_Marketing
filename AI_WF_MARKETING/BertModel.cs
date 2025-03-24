using Tensorflow;
using Tensorflow.NumPy;

namespace AI_WF_MARKETING
{
    public class BertModel
    {
        private readonly Session _session;
        private readonly Graph _graph;

        public BertModel(string modelDir)
        {
            _graph = new Graph().as_default();
            var bytes = File.ReadAllBytes(Path.Combine(modelDir, "saved_model.pb"));
            _graph.Import(bytes);
            _session = new Session(_graph);
        }

        public Tensor RunInference(Tensor inputIds, Tensor attentionMask, Tensor tokenTypeIds)
        {
            // Find the placeholder operations by name
            var inputIdsOp = _graph.OperationByName("serving_default_input_ids");
            var attentionMaskOp = _graph.OperationByName("serving_default_attention_mask");
            var tokenTypeIdsOp = _graph.OperationByName("serving_default_token_type_ids");
            var outputOp = _graph.OperationByName("StatefulPartitionedCall");

            var feedDict = new FeedItem[]
                                    {
                                        new FeedItem(inputIdsOp, inputIds),
                                        new FeedItem(attentionMaskOp, attentionMask),
                                        new FeedItem(tokenTypeIdsOp, tokenTypeIds)
                                    };

            // Call the session.run using a single output operation
            NDArray[] results = _session.run(
                                   new ITensorOrOperation[] { outputOp },
                                   feedDict
                               );

            return results[0];
        }
    }
}
