import numpy as np

class Layer:

    def __init__(self, size, connections):
        self.biases = np.random.rand(connections) * 2 - 1
        self.weights = np.random.rand(connections, size) * 2 - 1

    @staticmethod
    def sigmoid(value, deriv):
        if deriv:
            return value * (1 - value)
        return 1 / (1 + np.exp(-value))

    def get_output(self, input):
        self.activation = input
        self.output = Layer.sigmoid(self.weights.dot(input) + self.biases, False)
        return self.output

    def backprop(self, error):
        delta = error * Layer.sigmoid(self.output, True)
        a = 0.05 #learning rate
        self.weights -= np.outer(delta, self.activation) * a
        self.biases -= delta * a
        return np.dot(delta, self.weights)
