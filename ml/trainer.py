from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)
from datasets import load_dataset


from sklearn.model_selection import train_test_split
import pandas as pd
import time
import matplotlib.pyplot as plt
import sys, os
import torch


output_dir = f"sentiment_model-{time.strftime("%Y%m%d-%H%M%S")}"
os.makedirs(output_dir, exist_ok=True)


# Load your dataset
df = pd.read_csv("financial_phrasebank.csv")

# Check label distribution
print(df["label"].value_counts())
# 1    2535
# 2    1168
# 0     514

# Split into train (70%), temp (30%) -> then split temp into validation/test (50% each)
train, temp = train_test_split(df, test_size=0.3, random_state=42)
val, test = train_test_split(temp, test_size=0.5, random_state=42)

print(f"Train size: {len(train)}, Validation size: {len(val)}, Test size: {len(test)}")

# Save datasets
train.to_csv("train.csv", index=False)
val.to_csv("validation.csv", index=False)
test.to_csv("test.csv", index=False)

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(
    "NbAiLab/nb-bert-base", num_labels=3
)
tokenizer = AutoTokenizer.from_pretrained("NbAiLab/nb-bert-base")

# Load dataset
dataset = load_dataset(
    "csv",
    data_files={
        "train": "train.csv",
        "validation": "validation.csv",
        "test": "test.csv",
    },
)


# Tokenize data
def tokenize_function(examples):
    return tokenizer(
        examples["norwegian_sentence"],
        padding="max_length",
        truncation=True,
        max_length=256,
    )


tokenized_datasets = dataset.map(tokenize_function, batched=True)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=20,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=5,  # Avoid saving too many checkpoints
    load_best_model_at_end=True,
    metric_for_best_model="f1",
)


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }


# Train model
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model(output_dir)
trainer.evaluate()

df["label"].value_counts().plot(kind="bar")
plt.xlabel("Class")
plt.ylabel("Count")
plt.title("Class Distribution")
plt.savefig(f"{output_dir}/class-distribution.png")
plt.close()


loss_values = trainer.state.log_history
epochs = []
train_losses = []
val_epochs = []
val_losses = []

for entry in loss_values:
    if "epoch" in entry and "loss" in entry:
        epochs.append(entry["epoch"])
        train_losses.append(entry["loss"])
    if "epoch" in entry and "eval_loss" in entry:
        val_epochs.append(entry["epoch"])
        val_losses.append(entry["eval_loss"])

plt.plot(epochs, train_losses, marker="o", label="Training Loss")
plt.plot(val_epochs, val_losses, marker="x", label="Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Over Time")
plt.legend()
plt.savefig(f"{output_dir}/loss.png")
plt.close()

train_acc_epochs, train_accs = [], []
val_acc_epochs, val_accs = [], []
val_f1_epochs, val_f1s = [], []

for entry in loss_values:
    if "epoch" in entry and "accuracy" in entry:
        train_acc_epochs.append(entry["epoch"])
        train_accs.append(entry["accuracy"])
    if "epoch" in entry and "eval_accuracy" in entry:
        val_acc_epochs.append(entry["epoch"])
        val_accs.append(entry["eval_accuracy"])
    if "epoch" in entry and "eval_f1" in entry:
        val_f1_epochs.append(entry["epoch"])
        val_f1s.append(entry["eval_f1"])

plt.plot(val_acc_epochs, val_accs, marker="o", label="Validation Accuracy")
plt.plot(val_f1_epochs, val_f1s, marker="x", label="Validation F1")
plt.xlabel("Epochs")
plt.ylabel("Score")
plt.title("Validation Accuracy and F1 Over Time")
plt.legend()
plt.savefig(f"{output_dir}/val-acc-f1.png")


test_results = trainer.predict(tokenized_datasets["test"])

print("\n\nTest set Metrics:")
for key, value in test_results.metrics.items():
    print(f"{key}: {value}")


y_true = test_results.label_ids
y_pred = test_results.predictions.argmax(axis=-1)

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.title("Confusion Matrix")
plt.savefig(f"{output_dir}/confusion_matrix.png")
plt.close()
