import numpy as np
import time

class NeuralNetwork:
    def __init__(self, embeddings_size : int, vocabulary_size : int):
        self.embeddings_size = embeddings_size
        self.vocabulary_size = vocabulary_size

        self.w_input= np.random.randn(vocabulary_size, embeddings_size) * 0.01
        self.w_output = np.random.randn(embeddings_size, vocabulary_size) * 0.01

        self.grad_in_acc = np.zeros((vocabulary_size, embeddings_size))
        self.grad_out_acc = np.zeros((embeddings_size, vocabulary_size))

        self.counter = 0
        self.n_batch = 0
        self.loss_acc = 0
        self.first_loss = 0
        self.first_timestamp = 0

    def softmax(self, logits):
        exp_logits = np.exp(logits - np.max(logits))
        return exp_logits / np.sum(exp_logits)
        
    def foward(self, input_vector):
        input_hidden = np.mean(self.w_input[input_vector], axis = 0)
        logits = np.dot(input_hidden, self.w_output)
        return self.softmax(logits), input_hidden

    def multiclass_cross_entropy_loss(self, probabilities, target_id : int):
        return -np.log(probabilities[target_id] + 1e-9)
    
    def calculate_output_error(self, probabilities, target_id : int):
        probabilities.copy()
        probabilities[target_id] -= 1
        return probabilities
    
    def train_sample(self, input_vector, target_id : int, batch_size, learning_rate : float, terminal = False):
        self.counter += 1
        probabilities, hidden = self.foward(input_vector)
        self.loss_acc += self.multiclass_cross_entropy_loss(probabilities, target_id)
        output_error = self.calculate_output_error(probabilities, target_id)
        self.update_acc_gradients(output_error, hidden, input_vector)
        if self.counter >= batch_size:
            if terminal: self.print_terminal(batch_size, learning_rate)
            self.update_weights(learning_rate)

    def print_terminal(self, batch_size, learning_rate):
        loss = self.loss_acc / batch_size
        time_now = time.time()//1
        print("[Backpropagation learning results]")
        print("Batch"+f"(size:{batch_size})"+":", self.n_batch)
        print("Loss:", loss)
        if self.n_batch == 0:
            self.first_loss = loss
            self.first_timestamp = time_now
        else: 
            print("Diff with first batch loss:", loss - self.first_loss)
        print("Learning rate:", learning_rate)
        print("Timestamp:", time_now)
        print("Seconds pass since first batch:", time_now - self.first_timestamp)
        self.print_on_stats(f"{loss} {time_now}")

    def print_on_stats(self, message : str):
        with open("./traning_stats", "a") as log:
            log.write(f"{message}\n")
        
    def update_acc_gradients(self, output_error, hidden, input_vector):
        self.grad_in_acc[input_vector] += np.dot(self.w_output, output_error) / len(input_vector)
        self.grad_out_acc += np.outer(hidden, output_error)
    
    def update_weights(self, learning_rate : float):
        self.w_output -= learning_rate * (self.grad_out_acc / self.counter)
        self.w_input -= learning_rate * (self.grad_in_acc / self.counter)
        self.grad_in_acc = np.zeros((self.vocabulary_size, self.embeddings_size))
        self.grad_out_acc = np.zeros((self.embeddings_size, self.vocabulary_size))
        self.loss_acc = 0
        self.counter = 0
        self.n_batch += 1

    def save_weights(self, path : str):
        with open(f"{path}/w_in.npy", "wb") as w_in_file, \
             open(f"{path}/w_out.npy", "wb") as w_out_file:
            np.save(w_in_file, self.w_input)
            np.save(w_out_file, self.w_output)

    def load_weights(self, path : str):
        with open(f"{path}/w_in.npy", "rb") as w_in_file, \
             open(f"{path}/w_out.npy", "rb") as w_out_file:
            self.w_input = np.load(w_in_file)
            self.w_output = np.load(w_out_file)

    def end(self, learning_rate, batch_size):
        self.update_weights(learning_rate * (self.counter / batch_size))
    
    def prediction(self, input_vector : list) -> int:
        return np.argmax(self.foward(input_vector))


    