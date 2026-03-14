class Indexer:
    def __init__(self):
        self.idx2token = dict()
        self.token2idx = dict()
        self.unique_tokens = None

    def token_indexer(self, tokens : list):
        self.unique_tokens = list(set(tokens))
        with open("../Data/tokens_index", "w") as tokens_index:  
            for i in range(len(self.unique_tokens)):
                tokens_index.write(f"{self.unique_tokens[i]}\n")
                self.idx2token[i] = self.unique_tokens[i]
                self.token2idx[self.unique_tokens[i]] = i

    def index_loader(self, path : str):
        if not self.unique_tokens: self.unique_tokens = list()
        with open(path, "r") as tokens_index:
            i = 0
            for token in tokens_index.read().split('\n'):
                self.unique_tokens.append(i)
                self.idx2token[i] = token
                self.token2idx[token] = i
                i += 1
    
    def index_data(self, tokens : list) -> list:
        indexedData = list()
        for token in tokens:
            idx = self.getIndex(token)
            indexedData.append(idx)
        return indexedData


    def getIndex(self, token : str) -> int:
        return self.token2idx[token]
    
    def getToken(self, index : int) -> str:
        return self.idx2token[index]

    def getUniqueTokensList(self) -> list:
        return self.unique_tokens.copy()

if __name__ == "__main__":
    import Word2vec.src.preprocess as preprocess
    tokens = preprocess.load_data("/workspaces/JetBrains-Internship-Application/Word2vec/Data/cured_data", paragraph_split=False)
    basic_indexer = Indexer()
    basic_indexer.token_indexer(tokens)