# Intent Detection using XLM-Roberta

## Overview

This repository contains code and resources for building an Intent Detection model using the XLM-Roberta architecture. Intent detection is a common task in Natural Language Processing (NLP), where the goal is to determine the intent or purpose behind a user's input.

In this project, we leverage the power of the Hugging Face Transformers library to fine-tune an XLM-Roberta model for intent detection tasks. The model can be used for various applications, including chatbots, virtual assistants, and customer support systems.

## Features

- Preprocessing scripts for data preparation.
- Fine-tuning scripts for training the XLM-Roberta model on your dataset.
- Inference code for predicting intents using the trained model.
- Example notebooks and usage instructions.

## Getting Started

### Prerequisites

Before getting started, ensure you have the following prerequisites:

- Python 3.x
- PyTorch
- Hugging Face Transformers library
- Other dependencies as specified in `requirements.txt`

### Installation

1. Clone this repository to your local machine:

   ```shell
   git clone https://github.com/MLArtist/intent-detection-using-XLM-Roberta.git
   cd intent-detection-using-XLM-Roberta
   ```

2. Install the required dependencies:

   ```shell
   pip install -r requirements.txt
   ```

### Inferencing Instructions

To perform inference with the fine-tuned model, follow these steps:

1. Download the fine-tuned model configuration, weights, and label encoder indices from the [Google Drive Folder](https://drive.google.com/drive/folders/10PRchS1G8thw5kDZKBxjOwIZdLGlYAif).

2. Place the downloaded files in the 'xlm-roberta-large-custom-trained' folder.

Once you have completed these steps, run the inference server.

 ```shell
   python -m server --model xlm-roberta-large-custom-trained
   ```
3. test the inference server
   
   ```shell
   curl http://localhost:8080/ready
   curl -X POST -H "Content-Type: application/json" -d '{"text": "when is the next flight to new york"}' http://localhost:8080/intent
   ```

### Training Instructions

Check out the example notebook `Train.ipnb` in the `notebooks/` directory for detailed training and offline model evaluation.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)

## Authors

- [Amit Kumar](https://github.com/MLArtist)

For any inquiries or issues, please open an [issue](https://github.com/MLArtist/intent-detection-using-XLM-Roberta/issues).
