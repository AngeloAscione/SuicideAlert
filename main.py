import pandas as pd
from tensorflow.python.ops.gen_linalg_ops import batch_svd

import textutils
# from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import bert

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
    dataset = pd.read_csv("./dataset/cleaned_dataset.csv")
    dataset = dataset.reset_index(drop=True)
    # dataset = clean_data(dataset)
    # dataset = normalize_data(dataset)
    # dataset = clean_data(dataset)
    # encoder = LabelEncoder()
    # dataset['class'] = encoder.fit_transform(dataset['class'])
    # dataset.to_csv('./dataset/cleaned_dataset.csv', index=False)
    X_train, X_temp, y_train, y_temp = train_test_split(dataset['text'].reset_index(drop=True), dataset['class'].reset_index(drop=True), test_size=0.2, random_state=42)
    # Dividi temp in validation (10%) e test (10%)
    X_val, X_test, y_val, y_test = train_test_split(X_temp.reset_index(drop=True), y_temp.reset_index(drop=True), test_size=0.5, random_state=42)

    train_dataset = bert.TextDataset(X_train, y_train, bert.tokenizer)
    val_dataset = bert.TextDataset(X_val, y_val, bert.tokenizer)
    test_dataset = bert.TextDataset(X_test, y_test, bert.tokenizer)

    batch_size = 16
    train_loader = bert.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = bert.DataLoader(val_dataset, batch_size=batch_size)
    test_loader = bert.DataLoader(test_dataset, batch_size=batch_size)

    bert.train_model(3, train_loader)
    bert.evaluate_model(val_loader)
    bert.evaluate_model(test_loader)


if __name__ == "__main__":
    main()
