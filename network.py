import numpy as np
import os
import time
import math
from layer import Layer

class Network:

    def __init__(self, *layer_size):
        self.layer = [Layer(layer_size[i], layer_size[i + 1]) for i in range(len(layer_size) - 1)]

    def get_output(self, input):
        for layer in self.layer:
            input = layer.get_output(input)
        return input

    def get_prediction(self, model_output):
        return np.where(model_output == np.max(model_output))[0][0]

    def __backprop(self, input, output):
        model_output = self.get_output(input)
        prediction = self.get_prediction(model_output)
        error = model_output - output
        for layer in reversed(self.layer):
            error = layer.backprop(error)
        return output[prediction] == 1, np.square(model_output - output).sum()

    def train(self, epochs, inputs, outputs):
        start_time = time.time()
        loss = 0
        acc = 0
        b = 0.9999
        loss_track = []
        acc_track = []
        for epoch in range(epochs):
            k = 0
            for input, output in zip(inputs, outputs):
                k += 1
                b_acc, b_loss = self.__backprop(input, output)
                loss *= b
                if loss == 0:
                    loss = b_loss
                loss += b_loss * (1 - b)
                acc *= b
                if b_acc:
                    acc += 1 * (1 - b)
                if k % 100 == 0:
                    os.system("clear")
                    _acc = math.ceil(acc * 100) / 100
                    _loss = math.ceil(loss * 100) / 100
                    _time = math.ceil((time.time() - start_time) * 10) / 10
                    loss_track.append(loss)
                    acc_track.append(acc)
                    print(f"Epoch: {epoch} Time: {_time} Acc: {_acc} Loss: {_loss}")
        return acc_track, loss_track

