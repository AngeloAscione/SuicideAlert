import pandas as pd
import textutils
from sklearn.preprocessing import LabelEncoder

def print_initial_info(dataset):
    # print(dataset.head())
    # print(dataset.info())
    print(dataset.describe())
    #print(dataset.columns)
    print(dataset['class'].value_counts())

#Drops null rows and duplicates
def clean_data(dataset):
    dataset = dataset.dropna()
    dataset = dataset.drop_duplicates()
    return dataset

#Lower case for the text and removes punctuation and number, applies tokenization
def normalize_data(dataset):
    dataset['text'] = dataset['text'].apply(textutils.preprocess_text)
    return dataset

def main():
    dataset = pd.read_csv("./dataset/SuicideAndDepression_Detection.csv")
    dataset = clean_data(dataset)
    dataset = normalize_data(dataset)
    dataset = clean_data(dataset)
    encoder = LabelEncoder()
    dataset['class'] = encoder.fit_transform(dataset['class'])
    dataset.to_csv('./dataset/cleaned_dataset.csv', index=False)

    print(encoder.classes_)


if __name__ == "__main__":
    main()
