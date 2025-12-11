from transformers import DistilBertForSequenceClassification, Trainer, TrainingArguments
from data_loader import get_data

def train_contribution():
    train_dataset, test_dataset = get_data()

    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=4)

    training_args = TrainingArguments(
        output_dir="./results/contribution",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=3e-5,   # changed learning rate
        per_device_train_batch_size=64,  # bigger batch size
        per_device_eval_batch_size=64,
        num_train_epochs=5,   # more epochs
        weight_decay=0.05,    # stronger regularization
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
    model.save_pretrained("./results/contribution_model")

if __name__ == "__main__":
    train_contribution()
