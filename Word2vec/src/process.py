from collections import Counter
import re

# Eliminating all the titles from wikitext
def dataset_cleaning():
    with open("../Data/dataset", "r") as dataset, \
         open("../Data/no_labels_dataset", "w") as no_lable_dataset:
        for lines in dataset:
            if '=' not in lines:
                no_lable_dataset.write(lines)

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

def get_low_frequency_words(tokens : list, at_least = 2) -> list:
    word_count = Counter(tokens)
    return {word for word, count in word_count.items() if count <= at_least}

def remove_low_frequency_words(tokens : list, low_frequency_words : list) -> list:
    return [token for token in tokens if token not in low_frequency_words]

def tokenizer(text : str) -> list:
    tokens = tokenization(text)
    tokens = standarization(tokens)
    tokens = removing_stopwords(tokens)
    return tokens

def load_data(path : str, paragraph_split = True) -> list:
    with open(path, "r") as cured_data:
        if paragraph_split:
            data = list()
            for paragraph in cured_data.read().split('\n'):
                data.append(paragraph.split(' '))
            return data
        else:
            return cured_data.read().split()

def data_process():
    with open("../Data/no_labels_dataset", "r") as no_label_dataset, \
         open("../Data/standarized_data", "w") as standarized_data:
        for lines in no_label_dataset:
            tokens = tokenizer(lines)
            for token in tokens:
                standarized_data.write(f"{token} ")
            standarized_data.write('\n')

    tokens = load_data("../Data/standarized_data", paragraph_split=False)     
    low_frequency_words = get_low_frequency_words(tokens)
    with open("../Data/standarized_data", "r") as standarized_data, \
         open("../Data/cured_data", "w") as cured_data:
        for lines in standarized_data:
            tokens = remove_low_frequency_words(lines.split(), low_frequency_words)
            for token in tokens:
                cured_data.write(f"{token} ")
            cured_data.write('\n')
            


if __name__ == "__main__":
    dataset_cleaning()
    data_process()