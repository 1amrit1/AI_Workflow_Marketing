using System;
using System.Linq;
using Tensorflow;
using static Tensorflow.Binding;

namespace AI_WF_MARKETING
{
    class Program
    {
        static void Main(string[] args)
        {
            // 1. Instantiate the tokenizer
            var tokenizer = new BertTokenizerWrapper();

            // 2. Get user input
            Console.WriteLine("Enter text to process (for workflow or marketing):");
            string inputText = Console.ReadLine();

            // 3. Tokenize => (tokenIds, attentionMask, tokenTypeIds)
            var (tokenIds, attentionMask, tokenTypeIds) = tokenizer.Tokenize(inputText);

            // 4. Convert them to Tensors (shape [1, 256], int32).
            long[] shape = { 1, 256 };
            int[] idsInt = Array.ConvertAll(tokenIds, x => (int)x);
            int[] maskInt = Array.ConvertAll(attentionMask, x => (int)x);
            int[] typeInt = Array.ConvertAll(tokenTypeIds, x => (int)x);

            Tensor inputIdsTensor = tf.constant(idsInt, shape: shape, dtype: TF_DataType.TF_INT32);
            Tensor attentionMaskTensor = tf.constant(maskInt, shape: shape, dtype: TF_DataType.TF_INT32);
            Tensor tokenTypeIdsTensor = tf.constant(typeInt, shape: shape, dtype: TF_DataType.TF_INT32);

            // 5. Load the model
            var modelPath = "models/my_bert_savedmodel";
            var model = new BertModel(modelPath);

            // 6. Run inference
            Tensor output = model.RunInference(inputIdsTensor, attentionMaskTensor, tokenTypeIdsTensor);

            // 7. Print or interpret
            Console.WriteLine("\nModel Output:");
            Console.WriteLine(output.ToString());

            // 8. Human in the loop
            Console.WriteLine("\nAI Suggestion: [some logic based on output]");
            Console.Write("Do you approve? (y/n): ");
            string approval = Console.ReadLine();
            if (approval?.ToLower() == "y")
                Console.WriteLine("Decision Approved.");
            else
                Console.WriteLine("Decision Rejected.");

            Console.WriteLine("\nPress any key to exit.");
            Console.ReadKey();
        }
    }
}
