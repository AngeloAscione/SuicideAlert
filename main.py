import pandas as pd
import textutils
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

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
    # dataset = pd.read_csv("./dataset/SuicideAndDepression_Detection.csv")
    # dataset = clean_data(dataset)
    # dataset = normalize_data(dataset)
    # dataset = clean_data(dataset)
    # encoder = LabelEncoder()
    # dataset['class'] = encoder.fit_transform(dataset['class'])
    # dataset.to_csv('./dataset/cleaned_dataset.csv', index=False)

    dataset = pd.read_csv("./dataset/cleaned_dataset.csv")
    X = dataset['text']
    y = dataset['class']

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Training set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    print(f"Test set: {len(X_test)}")


if __name__ == "__main__":
    main()
