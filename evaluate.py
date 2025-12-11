import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

def evaluate_model(model_path="./results/contribution_model"):
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    # Load test dataset with pandas
    test_df = pd.read_json("hf://datasets/SetFit/ag_news/test.jsonl", lines=True)

    texts = test_df["text"].tolist()
    labels = test_df["label"].tolist()

    encodings = tokenizer(texts, truncation=True, padding=True, max_length=128, return_tensors="pt")

    with torch.no_grad():
        outputs = model(**encodings)
        preds = torch.argmax(outputs.logits, dim=1)

    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="macro")

    print(f"Accuracy: {acc:.4f}")
    print(f"F1 Score: {f1:.4f}")

    with open("results/metrics.txt", "w") as f:
        f.write(f"Model: {model_path}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    evaluate_model()
