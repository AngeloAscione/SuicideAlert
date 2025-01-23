import os
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Check device (use MPS for Apple Silicon if available)
print("Tento di caricare MPS")
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
print(device)

# Load Dataset
print("Carico il dataset")
csv_path = "dataset.csv"  # Update with your dataset path
data = pd.read_csv(csv_path)
print("Dataset caricato")


# Rename 'class' column to 'labels'
print("Rinomino colonne")
data.rename(columns={'class': 'labels'}, inplace=True)

# Train-Test Split
print("Splitto il dataset")
train_texts, val_texts, train_labels, val_labels = train_test_split(
    data['text'], data['labels'], test_size=0.2, random_state=42
)

# Tokenizer
print("Carico il modello")
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

def tokenize_function(texts):
    return tokenizer(
        texts.tolist(), padding=True, truncation=True, max_length=128, return_tensors="pt"
    )

# Tokenize Data
print("Tokenizzo il dataset di train")
train_encodings = tokenize_function(train_texts)
print("Tokenizzo il dataset di val")
val_encodings = tokenize_function(val_texts)

# Save tokenized datasets
print("Salvo i dataset tokenizzato")
tokenized_data_dir = "tokenized_data"
os.makedirs(tokenized_data_dir, exist_ok=True)
torch.save(train_encodings, os.path.join(tokenized_data_dir, "train_encodings.pt"))
torch.save(val_encodings, os.path.join(tokenized_data_dir, "val_encodings.pt"))
print(f"Tokenized datasets saved to {tokenized_data_dir}")

# Custom Dataset Class
class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels.iloc[idx])
        return item

train_dataset = CustomDataset(train_encodings, train_labels)
val_dataset = CustomDataset(val_encodings, val_labels)

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Model Setup
model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=data['labels'].nunique())

# Freeze all layers except for 9, 10, 11, and the classifier
for name, param in model.named_parameters():
    if not any(layer in name for layer in ["encoder.layer.9", "encoder.layer.10", "encoder.layer.11", "classifier"]):
        param.requires_grad = False

model = model.to(device)

# Optimizer and Loss
optimizer = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=5e-5)

# Checkpoint directory
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

# Training Loop
print("Inizio il training")
epochs = 3
for epoch in range(epochs):
    model.train()
    loop = tqdm(train_loader, leave=True)
    for batch in loop:
        # Move batch to device
        inputs = {key: val.to(device) for key, val in batch.items()}

        # Forward pass
        outputs = model(**inputs)
        loss = outputs.loss

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print loss
        loop.set_description(f"Epoch {epoch}")
        loop.set_postfix(loss=loss.item())

    # Save checkpoint after each epoch
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch}.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Evaluation
model.eval()
preds, true_labels = [], []
with torch.no_grad():
    for batch in val_loader:
        inputs = {key: val.to(device) for key, val in batch.items()}
        outputs = model(**inputs)
        logits = outputs.logits
        preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
        true_labels.extend(inputs['labels'].cpu().numpy())

accuracy = accuracy_score(true_labels, preds)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Save Model
output_dir = "bert_model_m1"
os.makedirs(output_dir, exist_ok=True)
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")
