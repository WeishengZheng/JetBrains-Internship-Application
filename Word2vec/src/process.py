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
    return [token for token in tokens if token not in {"in", "the", "is", "a", "\'s"}]

def tokenizer(text : str) -> list:
    tokens = tokenization(text)
    tokens = standarization(tokens)
    tokens = removing_stopwords(tokens)
    return tokens

def load_data(path : str, paragraph_split = True) -> list:
    with open(path, "r") as curedData:
        if paragraph_split:
            data = list()
            for paragraph in curedData.split('\n'):
                data.append(paragraph.split(' '))
        else:
            return curedData.read().split()

def data_process():
    with open("../Data/no_labels_dataset", "r") as noLabelDataSet, \
         open("../Data/cured_data", "w") as curedData:
        for lines in noLabelDataSet:
            tokens = tokenizer(lines)
            for token in tokens:
                curedData.write(f"{token} ")
            curedData.write('\n')


if __name__ == "__main__":
    dataset_cleaning()
    data_process()