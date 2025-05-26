from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
import numpy as np

model_name = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

labels = ['negative', 'neutral', 'positive']


def analyze_sentiment_transformer(text):

    # Tokenize and move inputs to the device
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}  # move all tensors to device

    with torch.no_grad():
        outputs = model(**inputs)

    scores = torch.nn.functional.softmax(outputs.logits, dim=1).cpu().numpy()[0]
    label = labels[np.argmax(scores)]
    return label, scores