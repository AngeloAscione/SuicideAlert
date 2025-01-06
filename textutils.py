import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('punkt_tab')

stop_words = set(stopwords.words('english'))
lemmetizer = WordNetLemmatizer()

def lower_text(text):
    return text.lower()

#Removes punctuation and numbers
def clean_text(text):
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub(r'\d+','',text)
    return text


#Applies lower case, clean text
def preprocess_text(text):
    #input("Prima: " + text)
    text = lower_text(text)
    text = clean_text(text)
    #input("Dopo: " + ' '.join(tokens))
    return text
