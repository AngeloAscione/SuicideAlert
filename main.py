import pandas as pd

df = pd.read_csv("dataset.csv")

# Esplora il dataset
print("Dataset originale:")
print(df.head())

# Pulizia del dataset
# Assumendo che il dataset abbia colonne "text" e "class" (etichette)
df = df.dropna(subset=["text", "class"]).reset_index(drop=True)  # Rimuovi righe con valori nulli

# Mappa le etichette in valori numerici
label_mapping = {"SuicideWatch": 0, "depression": 1, "teenagers": 2}
df["class"] = df["class"].map(label_mapping)

# Mostra il dataset pulito
print("Dataset pulito:")
print(df.head())

from datasets import Dataset

# Converte il DataFrame in un Dataset di Hugging Face
dataset = Dataset.from_pandas(df)

# Divide il dataset in training e test
dataset = dataset.train_test_split(test_size=0.2)
print("Training set:", len(dataset["train"]))
print("Test set:", len(dataset["test"]))

from transformers import BertTokenizer

# Carica il tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

# Funzione di tokenizzazione
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Tokenizza il dataset
# tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Rinomina la colonna "class" in "labels"
# tokenized_datasets = tokenized_datasets.rename_column("class", "labels")

# Imposta il formato come tensori PyTorch
# tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# tokenized_datasets.save_to_disk("./tokenized_datasets")
from datasets import DatasetDict
tokenized_datasets = DatasetDict.load_from_disk("./tokenized_datasets")
from transformers import BertForSequenceClassification

# Carica il modello pre-addestrato con 3 classi
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)
for name, param in model.named_parameters():
  if "encoder.layer.9" not in name and "encoder.layer.10" not in name and "encoder.layer.11" not in name and "classifier" not in name:
    param.requires_grad = False

from transformers import TrainingArguments, Trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Configura gli argomenti per l'addestramento
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
    load_best_model_at_end=True
)

# Definisci le metriche di valutazione
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"].shuffle(seed=42).select(range(8000)),
    eval_dataset=tokenized_datasets["test"].shuffle(seed=42).select(range(2000)),
    compute_metrics=compute_metrics
)

# Avvia l'addestramento
trainer.train()

# Valutazione
results = trainer.evaluate()
print("Risultati della valutazione:", results)

# Salva il modello e il tokenizer
model.eval()
prompt = "I feel so overwhelmed and hopeless about my future."
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=256)

import torch
# Passaggio 3: Effettua la predizione
with torch.no_grad():
    outputs = model(**inputs)

# Ottieni i logits (valori non normalizzati delle probabilità)
logits = outputs.logits

# Converti i logits nella classe predetta
predicted_class = torch.argmax(logits, dim=-1).item()

# Mostra il risultato
label_mapping_inverse = {0: "Suicide", 1: "Depression", 2: "Neenagers"}  # Etichette originali del dataset
predicted_label = label_mapping_inverse[predicted_class]
print(f"Testo: {prompt}")
print(f"Classe predetta: {predicted_label}")
