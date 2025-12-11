import pandas as pd
from transformers import DistilBertTokenizerFast
import torch
from torch.utils.data import Dataset

# Custom Dataset class for AG News
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

def get_data(tokenizer_name="distilbert-base-uncased", max_length=128):
    splits = {"train": "train.jsonl", "test": "test.jsonl"}

    # Load dataset with pandas
    train_df = pd.read_json("hf://datasets/SetFit/ag_news/" + splits["train"], lines=True)
    test_df = pd.read_json("hf://datasets/SetFit/ag_news/" + splits["test"], lines=True)

    tokenizer = DistilBertTokenizerFast.from_pretrained(tokenizer_name)

    train_dataset = AGNewsDataset(train_df, tokenizer, max_length)
    test_dataset = AGNewsDataset(test_df, tokenizer, max_length)

    return train_dataset, test_dataset
