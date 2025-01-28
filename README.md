# **SuicideAlert** 🛑  
**Un modello AI per l'identificazione di testi a rischio nei social media**  

## 📌 **Descrizione del Progetto**  
SuicideAlert è un sistema di classificazione testuale basato su **BERT** che mira a identificare e categorizzare contenuti testuali relativi al suicidio e alla depressione nei social media. Il progetto è stato sviluppato seguendo il framework **CRISP-DM** e sfrutta tecniche avanzate di fine-tuning per migliorare l'accuratezza della classificazione.  

## 🎯 **Obiettivo**  
L'obiettivo principale di SuicideAlert è automatizzare il riconoscimento di testi che potrebbero indicare situazioni di disagio mentale, fornendo un supporto per l’identificazione precoce di possibili segnali di allarme.  

## 📁 **Struttura del Repository**  
Il repository è organizzato come segue:  

```
SuicideAlert/
│── dataset/                     # Contiene il dataset utilizzato
│── models/                       # Modelli addestrati e pesi salvati
│── scripts/                      # Script per preprocessing, training e valutazione
│── results/                      # Risultati e metriche di valutazione
│── logs/                         # Log dell'addestramento
│── requirements.txt              # Dipendenze richieste
│── README.md                     # Documentazione del progetto
│── training.py                    # Script di addestramento del modello
│── evaluate.py                    # Script di valutazione del modello
│── preprocess.py                   # Script per pulizia e tokenizzazione del dataset
│── config.json                     # Configurazione del modello e dei parametri
```

---

## ⚙️ **Requisiti**  
Per eseguire questo progetto, assicurati di avere:  
- **Python 3.8+**  
- **Librerie necessarie:**  
  ```bash
  pip install -r requirements.txt
  ```

---

## 🚀 **Come replicare il progetto**  

### **1️⃣ Clonare il Repository**
```bash
git clone https://github.com/<username>/SuicideAlert.git
cd SuicideAlert
```

### **2️⃣ Preprocessare il Dataset**
Posiziona il dataset in formato `.csv` nella cartella `dataset/` e avvia il preprocessing:  
```bash
python preprocess.py
```

### **3️⃣ Addestrare il Modello**
```bash
python training.py
```
L'addestramento utilizza il modello **BERT** e congela progressivamente i layer per ottimizzare l'apprendimento.

### **4️⃣ Valutare il Modello**
```bash
python evaluate.py
```
Le metriche di valutazione includono **Accuracy, F1-Score, Precision e Recall**.

---

## 🔬 **Dettagli Implementativi**  
Il progetto utilizza il modello **BERT-base-uncased** con le seguenti impostazioni:  
- **Tokenizer** con padding e truncation  
- **Fine-Tuning con congelamento progressivo dei layer BERT**  
- **Training su 10 epoche** con *learning rate* di **2e-5**  
- **Valutazione con metriche multiple**  

### 📊 **Metriche del Modello**  
Dopo l'addestramento, il modello ha raggiunto le seguenti performance:  
- **Accuracy**: 86.2%  
- **F1-Score**: 86.2%  
- **Precision**: 86.3%  
- **Recall**: 86.2%  

---

## 📜 **Licenza**  
Questo progetto è distribuito sotto la **licenza MIT**.  

## 📩 **Contatti**  
Per suggerimenti o domande:  
✉️ **Nome Cognome** – [tuoemail@example.com](mailto:tuoemail@example.com)  
