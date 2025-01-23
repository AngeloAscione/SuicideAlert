import pandas as pd
from datasets import Dataset
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configurazione del dispositivo (Apple Silicon)
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if torch.backends.mps.is_available():
    print("Torch MPS è attivo")

# Caricamento e pulizia del dataset
df = pd.read_csv("dataset.csv")
print("Dataset originale:", df.head())

# Pulizia del dataset
df = df.dropna(subset=["text", "class"]).reset_index(drop=True)
label_mapping = {"SuicideWatch": 0, "depression": 1, "teenagers": 2}
df["class"] = df["class"].map(label_mapping)

# Conversione in Dataset di Hugging Face
dataset = Dataset.from_pandas(df)

# Divisione in train/test
dataset = dataset.train_test_split(test_size=0.2)
print(f"Train set: {len(dataset['train'])}, Test set: {len(dataset['test'])}")

# Caricamento del tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Tokenizzazione
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

tokenized_datasets = dataset.map(preprocess_function, batched=True)
tokenized_datasets = tokenized_datasets.rename_column("class", "labels")
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
tokenized_datasets.save_to_disk("./tokenized_datasets")

# Caricamento del modello
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
for name, param in model.named_parameters():
    if "encoder.layer.9" not in name and "encoder.layer.10" not in name and "encoder.layer.11" not in name and "classifier" not in name:
        param.requires_grad = False

model.to(device)

# Configurazione dell'addestramento
training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    save_total_limit=2,
    load_best_model_at_end=True,
)

# Metriche personalizzate
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    compute_metrics=compute_metrics,
)

# Avvio dell'addestramento
trainer.train()

# Valutazione
results = trainer.evaluate()
print("Risultati della valutazione:", results)

# Salvataggio del modello
model.save_pretrained("./bert_model")
tokenizer.save_pretrained("./bert_model")

# Inferenza
model.eval()
prompt = "I feel so overwhelmed and hopeless about my future."
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=256).to(device)

# Predizione
with torch.no_grad():
    outputs = model(**inputs)

logits = outputs.logits
predicted_class = torch.argmax(logits, dim=-1).item()

# Etichetta predetta
label_mapping_inverse = {0: "Suicide", 1: "Depression", 2: "Neutral"}
predicted_label = label_mapping_inverse[predicted_class]
print(f"Testo: {prompt}")
print(f"Classe predetta: {predicted_label}")
