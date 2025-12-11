import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import DistilBertTokenizerFast
from sklearn.metrics import accuracy_score, f1_score

# Custom Dataset for AG News
class AGNewsDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        text = str(self.dataframe.iloc[idx]["text"])
        label = int(self.dataframe.iloc[idx]["label"])

        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": torch.tensor(label, dtype=torch.long)
        }

# Simple MLP model
class MLPClassifier(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=256, num_classes=4):
        super(MLPClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def train_mlp():
    # Load dataset with pandas
    splits = {"train": "train.jsonl", "test": "test.jsonl"}
    train_df = pd.read_json("hf://datasets/SetFit/ag_news/" + splits["train"], lines=True)
    test_df = pd.read_json("hf://datasets/SetFit/ag_news/" + splits["test"], lines=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

    train_dataset = AGNewsDataset(train_df, tokenizer)
    test_dataset = AGNewsDataset(test_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)

    # MLP model
    model = MLPClassifier()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(5):
        model.train()
        for batch in train_loader:
            # Use [CLS] token embedding (first token)
            inputs = batch["input_ids"]
            labels = batch["labels"]

            # Convert token IDs to embeddings (simple average of IDs for baseline)
            embeddings = inputs.float()  # crude baseline: treat IDs as features
            outputs = model(embeddings)

            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} finished")

    # Evaluation
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in test_loader:
            inputs = batch["input_ids"]
            labels = batch["labels"]

            embeddings = inputs.float()
            outputs = model(embeddings)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.tolist())
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average="macro")

    print(f"MLP Accuracy: {acc:.4f}")
    print(f"MLP F1 Score: {f1:.4f}")

    with open("results/mlp_metrics.txt", "w") as f:
        f.write("MLP Baseline Results\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"F1 Score: {f1:.4f}\n")

if __name__ == "__main__":
    train_mlp()
