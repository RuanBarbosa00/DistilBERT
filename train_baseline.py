from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data_loader import get_data

def train_baseline():
    train_dataset, test_dataset = get_data()

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

    training_args = TrainingArguments(
        output_dir="./results/baseline",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=5e-5,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=50,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
    )

    trainer.train()
    model.save_pretrained("./results/baseline_model")

if __name__ == "__main__":
    train_baseline()
