import numpy as np
import matplotlib
from math import log
from NN import layer, NN

x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
# print(x.shape)

y = np.array([[0],
              [1],
              [1],
              [0]])

NN = NN(2, 16)

y_predicted = NN.calculate(x[0:1].T)
print(y_predicted)

NN.gradient_descent(1, 1000, x, y)

y_predicted = NN.calculate(x[0:1].T)
print(y_predicted)

# print(y[0, 0])
# print(cost(y[0, 0],y_predicted[0, 0] ))
