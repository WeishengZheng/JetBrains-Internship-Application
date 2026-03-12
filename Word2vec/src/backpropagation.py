import numpy as np

class NeuralNetwork:
    def __init__(self, embeddings_size : int, vocabulary_size : int):
        self.embeddings_size = embeddings_size
        self.vocabulary_size = vocabulary_size

        self.w_input= np.random.randn(vocabulary_size, embeddings_size) * 0.01
        self.w_output = np.random.randn(embeddings_size, vocabulary_size) * 0.01

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
        
    def foward(self, input_vector):
        input_hidden = np.mean(self.w_input[input_vector], axis = 0)
        logits = np.dot(input_hidden, self.w_output)
        return self.softmax(logits)

    def multiclass_cross_entropy_loss(self, probabilities, target_id : int):
        return -np.log(probabilities[target_id] + 1e-9)
    
    def calculate_output_error(self, probabilities, target_id : int):
        probabilities.copy()
        probabilities[target_id] -= 1
        return probabilities
    
    def update_weight(self, input_vector, output_error, learning_rate):
        gradient_w_in = np.dot(self.w_output, output_error)
        gradient_w_out = np.outer(self.w_input, output_error)

        self.w_output -= learning_rate * gradient_w_out
        n_inputs = len(input_vector)
        self.w_input[input_vector] -= learning_rate * (gradient_w_in / n_inputs)
    
    def prediction(self, input_vector : list) -> int:
        return np.argmax(self.foward(input_vector))


    