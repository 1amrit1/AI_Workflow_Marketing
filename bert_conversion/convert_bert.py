from transformers import TFBertModel

def main():
    model_name = "bert-base-uncased"  
    print(f"Loading the model: {model_name}")

    # Must have PyTorch installed for this to work
    model = TFBertModel.from_pretrained(model_name)

    output_dir = "./my_bert_savedmodel"
    print(f"Saving model to: {output_dir}")
    model.save_pretrained(output_dir, saved_model=True)

    print("Conversion complete!")

if __name__ == "__main__":
    main()