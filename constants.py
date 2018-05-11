# coding=utf-8
from numpy import array

# входной набор данных для обучения нейросети
INPUT = array([
    [0, 0, 0],
    [0, 0, 1],
    [0, 1, 0],
    [0, 1, 1],
    [1, 0, 0],
    [1, 0, 1],
    [1, 1, 0],
    [1, 1, 1]
])

# набор выходных данных для исходного входного набора AND
OUTPUT_AND = array([[0, 0, 0, 0, 0, 1, 0, 1]]).T

# набор выходных данных для исходного входного набора OR
OUTPUT_OR = array([[0, 1, 0, 1, 1, 1, 1, 1]]).T

