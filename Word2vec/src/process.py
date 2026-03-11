from collections import Counter
import re

# Eliminating all the titles from wikitext
def dataset_cleaning():
    with open("../Data/dataset", "r") as dataset, \
         open("../Data/no_labels_dataset", "w") as noLableDataset:
        for lines in dataset:
            if '=' not in lines:
                noLableDataset.write(lines)

def tokenization(text : str) -> list:
    text = text.split(" ")
    tokens = list()
    for token in text:
        if re.fullmatch(r"[a-zA-Z']+", token):
            tokens.append(token.lower())
        elif re.fullmatch(r"[0-9]+", token):
            tokens.append('#')
    return tokens

def standarization(tokens : list) -> list:
    return [token.lower() for token in tokens]

def removing_stopwords(tokens : list) -> list:
    return [token for token in tokens if token not in {"in", "the", "is", "a", "\'s", "\'"}]

def get_low_frequency_words(tokens : list, atLeast = 2) -> list:
    wordCount = Counter(tokens)
    return {word for word, count in wordCount.items() if count <= atLeast}

def remove_low_frequency_words(tokens : list, lowFrequencyWords : list) -> list:
    return [token for token in tokens if token not in lowFrequencyWords]

def tokenizer(text : str) -> list:
    tokens = tokenization(text)
    tokens = standarization(tokens)
    tokens = removing_stopwords(tokens)
    return tokens

def load_data(path : str, paragraph_split = True) -> list:
    with open(path, "r") as curedData:
        if paragraph_split:
            data = list()
            for paragraph in curedData.read().split('\n'):
                data.append(paragraph.split(' '))
            return data
        else:
            return curedData.read().split()

def data_process():
    with open("../Data/no_labels_dataset", "r") as noLabelDataSet, \
         open("../Data/standarized_data", "w") as standarizedData:
        for lines in noLabelDataSet:
            tokens = tokenizer(lines)
            for token in tokens:
                standarizedData.write(f"{token} ")
            standarizedData.write('\n')

    tokens = load_data("../Data/standarized_data", paragraph_split=False)     
    lowFrequencyWords = get_low_frequency_words(tokens)
    with open("../Data/standarized_data", "r") as standarizedData, \
         open("../Data/cured_data", "w") as curedData:
        for lines in standarizedData:
            tokens = remove_low_frequency_words(lines.split(), lowFrequencyWords)
            for token in tokens:
                curedData.write(f"{token} ")
            curedData.write('\n')
            


if __name__ == "__main__":
    dataset_cleaning()
    data_process()