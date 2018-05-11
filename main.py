# coding=utf-8
from constants import *
from neural_network import NeuralNetwork
from neural_layer import NeuralLayer
import numpy as np


# def convert(f):
#     if f > 0.75:
#         return 1
#     else:
#         return 0


if __name__ == "__main__":
    np.random.seed(1)

    # создание нейросети
    neural_network = NeuralNetwork(
        [
            NeuralLayer(4, 3),
            NeuralLayer(5, 4),
            NeuralLayer(4, 5),
            NeuralLayer(1, 4),
        ],
        0.7
    )

    a = 1
    c = 0

    neural_network.train(INPUT, OUTPUT_AND, 10000)

    print "{0} AND {1} -> {2}".format(a, c, neural_network.process(array([a, 1, c]))[0])

    neural_network.train(INPUT, OUTPUT_OR, 10000)
    print "{0} OR  {1} -> {2}".format(a, c, neural_network.process(array([a, 1, c]))[0])

    # neural_network.print_structure()

