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
        print "\n Layer {0} of {1} neurons structure:".format(i, self.number_of_neurons)
        for j in xrange(self.number_of_neurons):
            print "  Neuron {0} input weights:".format(j)
            for k in xrange(len(self.synaptic_weights)):
                print "   Weight {0} = {1}".format(k, self.synaptic_weights[k][j])
