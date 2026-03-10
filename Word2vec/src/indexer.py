import process

class indexer:
    def __init__(self):
        self.idx2token = dict()
        self.token2idx = dict()

    def token_indexer(self, tokens : list):
        unique_tokens = list(set(tokens))
        with open("../Data/tokens_index", "w") as tokens_index:  
            for i in range(len(unique_tokens)):
                tokens_index.write(f"{unique_tokens[i]}\n")
                self.idx2token[i] = unique_tokens[i]
                self.token2idx[unique_tokens[i]] = i

    def index_loader(self, path : str):
        with open(path, "r") as tokens_index:
            i = 0
            for token in tokens_index.read().split('\n'):
                self.idx2token[i] = token
                self.token2idx[token] = i
                i += 1

    def getIndex(self, token : str) -> int:
        return self.token2idx[token]
    
    def getToken(self, index : int) -> str:
        return self.idx2token[index]

if __name__ == "__main__":
    #tokens = process.load_tokenizer("/workspaces/JetBrains-Internship-Application/Word2vec/Data/cured_data")
    basic_indexer = indexer()
    #basic_indexer.token_indexer(tokens)
    basic_indexer.index_loader("/workspaces/JetBrains-Internship-Application/Word2vec/Data/tokens_index")
    print(f"Index of hello: {basic_indexer.getIndex("dvorak")}")
    print(f"dvorak == {basic_indexer.getToken(11580)}")