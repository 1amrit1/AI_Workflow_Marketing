using System;
using System.Linq;
using BERTTokenizers;

namespace AI_WF_MARKETING
{
    public class BertTokenizerWrapper
    {
        private readonly BertBaseTokenizer _tokenizer;

        public BertTokenizerWrapper()
        {
            // Using the default base_cased.txt in ./Vocabularies
            _tokenizer = new BertBaseTokenizer();
        }

        public (long[] tokenIds, long[] attentionMask, long[] tokenTypeIds) Tokenize(string text)
        {
            var encoded = _tokenizer.Encode(256, text);
            var inputIds = encoded.Select(t => t.InputIds).ToArray();
            var attMask = encoded.Select(t => t.AttentionMask).ToArray();
            var tokenTypeIds = encoded.Select(t => t.TokenTypeIds).ToArray();
            return (inputIds, attMask, tokenTypeIds);
        }
    }
}
