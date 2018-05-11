# coding=utf-8
from numpy import exp, dot


class NeuralNetwork:
    def __init__(self, layers, train_speed):
        self.layers = layers
        self.train_speed = train_speed

    # функция активации
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + exp(-x))

    # производная функции активации
    @staticmethod
    def sigmoid_derivative(x):
        return x * (1 - x)

    # обучение нейросети
    def train(self, training_set_inputs, training_set_outputs, number_of_training_iterations):

        # итерация по входным наборам данных
        for iteration in xrange(number_of_training_iterations):

            # прохождение очередного набора входных данных через нейросеть для получения значений на выходах
            self.process(training_set_inputs)

            # нахождение ошибок весов синапсов
            for i in xrange(len(self.layers)):
                j = len(self.layers) - i - 1

                if j == len(self.layers) - 1:

                    self.layers[j].layer_errors = training_set_outputs - self.layers[j].layer_outputs
                    self.layers[j].layer_deltas = self.layers[j].layer_errors

                else:

                    self.layers[j].layer_errors = self.layers[j + 1].layer_deltas.dot(
                        self.layers[j + 1].synaptic_weights.T)

                    self.layers[j].layer_deltas = self.layers[j].layer_errors * self.sigmoid_derivative(
                        self.layers[j].layer_outputs)

            # корректировка весов синапсов
            for i in xrange(len(self.layers)):
                if i == 0:
                    adjustment = training_set_inputs.T.dot(self.layers[i].layer_deltas)
                else:
                    adjustment = self.layers[i - 1].layer_outputs.T.dot(self.layers[i].layer_deltas)

                self.layers[i].synaptic_weights += adjustment * self.train_speed

    def print_structure(self):
        print "Layers count: {0}".format(len(self.layers))
        for i in xrange(len(self.layers)):
            self.layers[i].print_structure(i)

    # получение значений на выходах нейронов
    def process(self, inputs):
        for i in xrange(len(self.layers)):
            if i == 0:
                self.layers[i].layer_outputs = self.sigmoid(dot(inputs, self.layers[i].synaptic_weights))
            else:
                self.layers[i].layer_outputs = self.sigmoid(
                    dot(self.layers[i - 1].layer_outputs, self.layers[i].synaptic_weights))

        return self.layers[len(self.layers) - 1].layer_outputs
