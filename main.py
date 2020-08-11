import numpy as np
from network import Network
import matplotlib.pyplot as plt

def load_training_data():
    inputs = np.genfromtxt("train.csv", delimiter=",", dtype=int, max_rows=50000)
    outputs = []
    for i in range(len(inputs)):
        vector = inputs[i]
        output = np.zeros(10)
        output[vector[0]] = 1
        outputs.append(output)
    inputs = np.delete(inputs, 0, 1) / 255 #inputs between 0 and 1
    return inputs, outputs

def init_network():
    global network, inputs, outputs
    network = Network(784, 16, 16, 10)
    inputs, outputs = load_training_data()

init_network()
acc, loss = network.train(3, inputs, outputs)

fig, ax = plt.subplots(figsize=(4, 3))

ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")
ax.xaxis.set_ticks_position("bottom")
ax.yaxis.set_ticks_position("left")

ax.plot(acc, color="#00FF00", label="Accuracy%")
ax.plot(loss, color="#FF0000", label="Loss")
ax.legend(loc="upper right", frameon=False)
plt.show()
