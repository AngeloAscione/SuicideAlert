import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# Verifica il dispositivo disponibile
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
if torch.backends.mps.is_available():
    print("Torch è configurato per utilizzare il backend MPS su Apple Silicon.")

# Carica il modello e il tokenizer
model_dir = "./model/bert_model_validation" 
if not os.path.exists(model_dir):
    raise FileNotFoundError(f"La directory del modello non esiste: {model_dir}")

print(f"Caricamento del modello da {model_dir}...")
model = BertForSequenceClassification.from_pretrained(model_dir)
tokenizer = BertTokenizer.from_pretrained(model_dir)
model.to(device)
model.eval()

print("\nShell interattiva per testare il modello. Digita 'exit' per uscire.")
while True:
    prompt = input("Inserisci un testo da analizzare: ")
    if prompt.lower() == "exit":
        print("Uscita dalla shell interattiva.")
        break

    # Tokenizza l'input
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=256)
    inputs = {key: val.to(device) for key, val in inputs.items()}

    # Ottieni la predizione
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=-1).item()

    # Mappa la predizione alla classe originale
    label_mapping = {0: "Suicide", 1: "Depression", 2: "Neutral"} 
    predicted_label = label_mapping.get(predicted_class, "Unknown")

    print(f"Testo: {prompt}")
    print(f"Classe predetta: {predicted_label}\n")
