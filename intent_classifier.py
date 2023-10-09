# -*- coding: utf-8 -*-
from roberta import Roberta


class IntentClassifier:
    def __init__(self):
        self.model_loaded = False
        self.model = Roberta()

    def is_ready(self):
        return self.model_loaded if self.model_loaded else False

    def load(self, file_path):
        self.model_loaded = self.model.load(file_path)

    def classify_intent_function(self, text):
        return self.model.inference(text)


if __name__ == '__main__':
    model = IntentClassifier()
    model.load("xlm-roberta-large-custom-trained")
    print(model.is_ready())
