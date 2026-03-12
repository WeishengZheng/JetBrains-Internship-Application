import indexer
import process
import numpy as np

class Word2vec:
    def __init__(self, window = 2, embeddings = 300):
        self.window = window
        self.embeddings = embeddings
        self.weight_matrix = None
        self.idxr = None

    def _create_weight_matrix(self, unique_tokens : list):
        self.weight_matrix = np.random.rand(len(unique_tokens), self.embeddings)

    def add_vocabulary(self, idxr : indexer.Indexer):
        self.idxr = idxr
        self._create_weight_matrix(idxr.getUniqueTokensList())

    def training(self, training_material):
        if self.idxr == None: 
            print("There is no indexer or weight_matrix in the model")
        for p in training_material:
            paragraph = list()
            for tokens in p:
                paragraph.append(self.idxr.getIndex(tokens))
            paragraph = np.array(paragraph, np.int32)

            for i in range(self.window, len(paragraph) - self.window):
                input_vector = np.array([paragraph[j+i-self.window] for j in range(self.window*2 + 1) if j != self.window], np.int32)
                output_Vector = np.array([paragraph[i]], np.int32)




if __name__ == "__main__":
    word2vec = Word2vec()
    basic_indexer = indexer.Indexer()
    basic_indexer.index_loader("/workspaces/JetBrains-Internship-Application/Word2vec/Data/tokens_index")
    word2vec.add_vocabulary(basic_indexer)
    word2vec.training(process.load_data("/workspaces/JetBrains-Internship-Application/Word2vec/Data/cured_data"))
