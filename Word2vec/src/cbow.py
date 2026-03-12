import indexer
import process
import backpropagation as bp
import numpy as np

class Word2vec:
    def __init__(self, window = 2, embeddings = 300):
        self.window = window
        self.embeddings = embeddings
        self.idxr = None
        self.neural_network = None

    def _create_weight_matrix(self, unique_tokens : list):
        self.neural_network = bp.NeuralNetwork(self.embeddings, len(unique_tokens))

    def add_vocabulary(self, idxr : indexer.Indexer):
        self.idxr = idxr
        self._create_weight_matrix(idxr.getUniqueTokensList())

    def training(self, training_material, batch_size, learning_rate, epoach):
        for p in training_material:
            paragraph = list()
            for tokens in p:
                paragraph.append(self.idxr.getIndex(tokens))
            paragraph = np.array(paragraph, np.int32)

            for i in range(self.window, len(paragraph) - self.window):
                input_vector = np.array([paragraph[j+i-self.window] for j in range(self.window*2 + 1) if j != self.window], np.int32)
                output_vector = np.array([paragraph[i]], np.int32)
                self.neural_network.train_sample(input_vector, output_vector[0], batch_size, learning_rate, terminal = True)
        self.neural_network.end(batch_size, learning_rate)

if __name__ == "__main__":
    word2vec = Word2vec()
    basic_indexer = indexer.Indexer()
    basic_indexer.index_loader("/workspaces/JetBrains-Internship-Application/Word2vec/Data/tokens_index")
    word2vec.add_vocabulary(basic_indexer)
    word2vec.training(process.load_data("/workspaces/JetBrains-Internship-Application/Word2vec/Data/cured_data"), 32, 0.05, 0)
