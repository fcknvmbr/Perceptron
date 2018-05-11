# coding=utf-8
import numpy as np


class NeuralLayer:

    def __init__(self, number_of_neurons, number_of_inputs_per_neuron):

        self.number_of_neurons = number_of_neurons
        self.number_of_inputs_per_neuron = number_of_inputs_per_neuron

        # рандомное заполнение весов синапсов для нейронов
        self.synaptic_weights = 2 * np.random.random((number_of_inputs_per_neuron, number_of_neurons)) - 1

        # ошибки для каждого из синапсов
        self.layer_errors = np.array([])

        # дельта для подсчета ошибки каждого из синапсов
        self.layer_deltas = np.array([])

        # значение на выходе для нейрона при каждом из наборов комбинаций синапсов
        self.layer_outputs = np.array([])

    def print_structure(self, i):
        print(f"\n Layer {i} of {self.number_of_neurons} neurons structure:")
        for j in range(self.number_of_neurons):
            print(f"  Neuron {j} input weights:")
            for k in range(len(self.synaptic_weights)):
                print(f"   Weight {k} = {self.synaptic_weights[k][j]}")
