from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch, joblib
import numpy as np
import json

with open ('temprature.txt', 'r') as f:
    loaded_temprature = float(f.read())


tokenizer = AutoTokenizer.from_pretrained('artifacts/distilbert_intents')
model = AutoModelForSequenceClassification.from_pretrained('artifacts/distilbert_intents')
tfidf = joblib.load("artifacts/tfidf_baseline.joblib")
message = 'Can you track my order'

tokenized_message = tokenizer(message, return_tensors='pt')
print("Input:", [message])
output_tfidf = tfidf.predict([message])
print(output_tfidf)
with torch.no_grad():
    
    model.eval()
    output = model(**tokenized_message)
    softmaxed = softmax(output.logits / loaded_temprature)

confidence = np.max(softmaxed, axis=1)[0]
index_location = np.argmax(softmaxed, axis=1)[0]

print(f'The highest value is {confidence} in index {index_location} that has intent {model.config.id2label[index_location]}')

if np.max(softmaxed, axis=1) <= 0.9:
    dictionary = {
        "Ticket ID": "ticket-001",
        "Message": message,
        "Suggested Intent": model.config.id2label[index_location],
        "Confidence": float(confidence),
        "Review reason": "confidence at or below threshold"
    }
    print(dictionary)
    with open('review_queue.jsonl', 'a') as f:
        f.write(json.dumps(dictionary) + "\n")

else:
    print(model.config.id2label[index_location])



