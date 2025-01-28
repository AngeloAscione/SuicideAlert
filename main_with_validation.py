import pandas as pd
from datasets import Dataset, DatasetDict, load_from_disk
from transformers import BertTokenizer, BertForSequenceClassification, TrainingArguments, Trainer
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import os

# Imposta il dispositivo
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if torch.backends.mps.is_available():
    print("Torch su MPS disponibile")

'''
# Carica il dataset
df = pd.read_csv("dataset.csv")

# Pulizia del dataset
df = df.dropna(subset=["text", "class"]).reset_index(drop=True)
label_mapping = {"SuicideWatch": 0, "depression": 1, "teenagers": 2}
df["class"] = df["class"].map(label_mapping)

# Converte il DataFrame in un Dataset di Hugging Face
dataset = Dataset.from_pandas(df)

# Suddivide il dataset in training, validation e test
temp_dataset = dataset.train_test_split(test_size=0.2)
train_validation_split = temp_dataset["train"].train_test_split(test_size=0.1)

dataset = {
    "train": train_validation_split["train"],
    "validation": train_validation_split["test"],
    "test": temp_dataset["test"]
}

# Carica il tokenizer
'''
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
'''
# Funzione di tokenizzazione
def preprocess_function(examples):
    return tokenizer(examples["text"], truncation=True, padding="max_length", max_length=256)

# Tokenizza il dataset
tokenized_datasets = {split: ds.map(preprocess_function, batched=True) for split, ds in dataset.items()}
for split in tokenized_datasets:
    tokenized_datasets[split] = tokenized_datasets[split].rename_column("class", "labels")
    tokenized_datasets[split].set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# Salva i dataset tokenizzati
tokenized_datasets["train"].save_to_disk("./tokenized_train")
tokenized_datasets["validation"].save_to_disk("./tokenized_validation")
tokenized_datasets["test"].save_to_disk("./tokenized_test")
'''

tokenized_datasets = dict()
tokenized_datasets["train"] = load_from_disk("./tokenized_train")
tokenized_datasets["validation"] = load_from_disk("./tokenized_validation")
tokenized_datasets["test"] = load_from_disk("./tokenized_test")
# Carica il modello
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=3)

# Congela i primi 8 layer inizialmente
for name, param in model.named_parameters():
    print(name)
    if "encoder.layer." in name:
        try:
            layer_num = int(name.split(".")[3])
            if layer_num <= 8:
                param.requires_grad = False
            else:
                print(f"Layer {layer_num} inizialmente sbloccato.")
        except ValueError:
            continue

model.to(device)

save_dir = "./results_second_model"

# Configura gli argomenti per l'addestramento
training_args = TrainingArguments(
    output_dir=save_dir,
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=3,  # Mantieni solo gli ultimi 3 checkpoint
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir="./logs",
    logging_steps=100,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy"
)

# Funzione per calcolare le metriche
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted")
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "f1": f1, "precision": precision, "recall": recall}

# Congelamento progressivo dei layer
def unfreeze_layers(epoch):
    if epoch == 3 :
        for name, param in model.named_parameters():
            print(name)
            if "encoder.layer." in name:
                try:
                    layer_num = int(name.split(".")[3])
                    if 6 < layer_num <= 8:
                        param.requires_grad = True
                        print(f"Layer {layer_num} sbloccato all'epoca 10.")
                except ValueError:
                    continue
    elif epoch == 6:
        for name, param in model.named_parameters():
            print(name)
            if "encoder.layer." in name:
                try:
                    layer_num = int(name.split(".")[3])
                    if layer_num == 6 :
                        param.requires_grad = True
                        print(f"Layer {layer_num} sbloccato all'epoca 20.")
                except ValueError:
                    continue

# Callbacks per congelamento dinamico
class CustomTrainer(Trainer):
    def training_step(self, *args, **kwargs):
        unfreeze_layers(self.state.epoch)
        return super().training_step(*args, **kwargs)


last_checkpoint = None
if os.path.isdir(save_dir):
    checkpoints = [os.path.join(save_dir, d) for d in os.listdir(save_dir) if d.startswith("checkpoint")]
    if checkpoints:
        last_checkpoint = max(checkpoints, key=os.path.getmtime)  # Ottieni il checkpoint più recente

if last_checkpoint:
    print(f"Trovato un checkpoint: {last_checkpoint}. Riprendo l'addestramento da qui.")
else:
    print("Nessun checkpoint trovato. Inizio un nuovo addestramento.")

# Usa il trainer personalizzato
custom_trainer = CustomTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
    compute_metrics=compute_metrics
)

# Avvia l'addestramento
custom_trainer.train(resume_from_checkpoint=last_checkpoint)

# Valutazione finale
results = custom_trainer.evaluate(tokenized_datasets["test"])
print("Risultati sul test set:", results)

# Salva il modello e il tokenizer
model.save_pretrained("./bert_model_validation")
tokenizer.save_pretrained("./bert_model_validation")
