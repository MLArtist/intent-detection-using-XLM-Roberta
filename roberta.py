import torch
import pickle
import os
from transformers import XLMRobertaTokenizer, XLMRobertaForSequenceClassification

# Check if CUDA (GPU) is available, otherwise use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Define a class for working with the XLM-Roberta model
class Roberta:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.label_encoder = None

    def load(self, model_path):
        # Check if model files exist at the provided path
        if os.path.exists(os.path.join(model_path, "model.safetensors")):
            print(f"Loading model from local path: {model_path}")

            model_name = "-".join(model_path.split("-")[:3])  # Extract model name from path

            # Load model and tokenizer
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            self.model = XLMRobertaForSequenceClassification.from_pretrained(model_path)

        else:
            model_name = "-".join(model_path.split("-")[:3])  # Extract model name from path
            print(f"Model files not found at {model_path}. Downloading from Hugging Face...")

            # Download model and tokenizer directly
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(model_name)
            self.model = XLMRobertaForSequenceClassification.from_pretrained(model_name)

            # Save the downloaded model to the specified path
            self.model.save_pretrained(model_path)

        self.model.to(device)

        # Load the label encoder from the file
        self.label_encoder = pickle.load(open(model_path + "/label_encoder.pkl", "rb"))
        return True

    def inference(self, text, k=3):
        # Tokenize the input text
        encoded_text = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")
        encoded_text.to(device)

        # Perform inference
        with torch.no_grad():
            outputs = self.model(encoded_text.input_ids.to(device),
                                 attention_mask=encoded_text.attention_mask.to(device))
        predicted_probabilities = torch.softmax(outputs.logits, dim=1)

        # Get the top three predicted classes and their corresponding probabilities
        k = k if k <= predicted_probabilities.shape[1] else predicted_probabilities.shape[1]
        top_classes = torch.topk(predicted_probabilities, k, dim=1)
        top_class_indices = top_classes.indices[0].tolist()
        top_class_probabilities = top_classes.values[0].tolist()

        # Map the class indices back to the original string labels using the label encoder
        top_class_labels = self.label_encoder.inverse_transform(top_class_indices)

        res = []

        # Prepare the result as a list of dictionaries
        for label, probability in zip(top_class_labels, top_class_probabilities):
            res.append({"label": label, "confidence": f"{probability:.4f}"})
        return res


if __name__ == '__main__':
    input_text = "suggest cheapest flight tickets"
    model = Roberta()
    model.load("xlm-roberta-large-custom-trained")
    print(model.inference(input_text))
