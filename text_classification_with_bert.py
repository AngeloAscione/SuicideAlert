from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Carica il modello e il tokenizer salvati
MODEL_PATH = "./bert_model"  # Percorso dove è stato salvato il modello
tokenizer = BertTokenizer.from_pretrained(MODEL_PATH)
model = BertForSequenceClassification.from_pretrained(MODEL_PATH)
device = torch.device("mps")
model.to(device)

# Mappa delle etichette
label_mapping_inverse = {
    0: "Suicide",
    1: "Depression",
    2: "Neutral",
}


# Funzione per fare una predizione
def classify_text(prompt):
    # Prepara l'input per il modello
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding="max_length",
        max_length=256
    )

    # Disattiva il calcolo dei gradienti
    with torch.no_grad():
        outputs = model(**inputs)

    # Ottieni i logits e calcola la classe predetta
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=-1).item()
    predicted_label = label_mapping_inverse[predicted_class]

    return predicted_label


if __name__ == "__main__":
    print("Inserisci un testo per la classificazione (digita 'exit' per uscire):")
    while True:
        # Input manuale dell'utente
        prompt = input(">> ")
        if prompt.lower() == "exit":
            break

        # Classifica il testo con il modello
        predicted_label = classify_text(prompt)

        # Mostra la classe predetta
        print(f"Classe predetta: {predicted_label}")