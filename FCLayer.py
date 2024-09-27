from numba import cuda, jit
import numpy as np

class FullyConnectedLayer:
    def __init__(self, input_size, output_size, learning_rate=0.01):
        self.weights = np.random.uniform(0, 1, (output_size, input_size)).astype(np.float32)
        self.biases = np.zeros(output_size, dtype=np.float32)
        self.learning_rate = learning_rate

    def forward(self, input_vector):
        self.input = input_vector
        self.output = np.dot(self.weights, self.input) + self.biases
        
        self.probabilities = softmax(self.output)
        return self.probabilities

    def backward(self, target):
        # Calculate error (difference between probabilities and target)
        self.error = self.probabilities - target

        self.weights -= self.learning_rate * np.outer(self.error, self.input)
        self.biases -= self.learning_rate * self.error

def softmax(x):
    exp_x = np.exp(x - np.max(x))  # Numerical stability
    return exp_x / exp_x.sum()