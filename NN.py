import numpy as np
import matplotlib
from math import log


class NN():
    def __init__(self,  input_size, layer1_size):
        self.input_size = input_size
        self.layer1_size = layer1_size

        self.layer_1 = layer(input_size, layer1_size)
        self.layer_2 = layer(layer1_size, 1)

        self.layer_1_update = layer(input_size, layer1_size)
        self.layer_2_update = layer(layer1_size, 1)

    def calculate(self, x):
        y_predicted = self.layer_2.calculate(self.layer_1.calculate(x))
        return float(y_predicted[0])

    def gradient_descent(self, a, epochs, x,  y):
        for i in range(epochs):
            y_pred = np.ones((y.shape[0], y.shape[1]), dtype=float)
            for k in range(y_pred.shape[0]):
                y_pred[k, 0] = self.calculate(x[k:k+1].T)
            print(y)
            print(y_pred)
            self.update(a, x, y)
            self.layer_1.b = self.layer_1_update.b
            self.layer_1.w = self.layer_1_update.w
            self.layer_2.b = self.layer_2_update.b
            self.layer_2.w = self.layer_2_update.w

            cost = 0
            for j in range(y.shape[0]):
                cost = cost + loss(y[j, 0], y_pred[j, 0])
            cost = cost/y.shape[0]
            print('cost after epoch %i is %f' % (i, cost))

    def update(self, a, X, Y):

        denom = Y.shape[0]
        self.layer_1_update.set_zero()
        self.layer_2_update.set_zero()

        for k in range(denom):
            # print(X)
            x = X[k:k+1].T
            # print(x)
            y = Y[k, 0]
            y_pred = self.layer_2.calculate(self.layer_1.calculate(x))
            self.layer_2_update.b[0, 0] = self.layer_2_update.b[0,
                                                                0] + self.dL_db2(y, y_pred)

            for i in range(self.layer_2.w.shape[0]):
                self.layer_2_update.w[i, 0] = self.layer_2_update.w[i,
                                                                    0] + self.dL_dw2(y, y_pred, i)

            for i in range(self.layer_1.b.shape[0]):
                self.layer_1_update.b[i, 0] = self.layer_1_update.b[i,
                                                                    0] + self.dL_db1(y, y_pred, i)

            for i in range(self.layer_1.w.shape[1]):
                for j in range(self.layer_1.w.shape[0]):
                    self.layer_1_update.w[j, i] = self.layer_1_update.w[j,
                                                                        i] + self.dL_dw1(x, y, y_pred, i, j)

        self.layer_2_update.b[0, 0] = self.layer_2.b[0,
                                                     0] - a * self.layer_2_update.b[0, 0]/denom
        for i in range(self.layer_2.w.shape[0]):
            self.layer_2_update.w[i, 0] = self.layer_2.w[i,
                                                         0] - a * self.layer_2_update.w[i, 0]/denom
        for i in range(self.layer_1.b.shape[0]):
            self.layer_1_update.b[i, 0] = self.layer_1.b[i,
                                                         0] - a * self.layer_1_update.b[i, 0]/denom
        for i in range(self.layer_1.w.shape[1]):
            for j in range(self.layer_1.w.shape[0]):
                self.layer_1_update.w[j, i] = self.layer_1.w[j,
                                                             i] - a * self.layer_1_update.w[j, i]/denom

    def dL_dw1(self, x, y, y_pred,  i, j):
        dL_da2 = dL_dy(y, y_pred)
        da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
        dz2_da1 = self.layer_2.w[i, 0]  # a1 = layer1.a[j, 0]
        da1_dw1 = sigmoid_dash(self.layer_1.z[i, 0])
        dw1_dw1 = x[j, 0]
        dL_dw1 = dL_da2 * da2_dz2 * dz2_da1 * da1_dw1 * dw1_dw1
        return dL_dw1

    def dL_dw2(self, y, y_pred,  j):
        dL_da2 = dL_dy(y, y_pred)
        da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
        dz2_dw2 = self.layer_1.a[j, 0]  # w2 = layer2.w[j, 0]
        dL_dw2 = dL_da2 * da2_dz2 * dz2_dw2
        return dL_dw2

    def dL_db1(self, y, y_pred,  j):
        dL_da2 = dL_dy(y, y_pred)
        da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
        dz2_da1 = self.layer_2.w[j, 0]  # a1 = layer1.a[j, 0]
        da1_db1 = self.layer_1.b[j, 0]
        dL_db1 = dL_da2 * da2_dz2 * dz2_da1 * da1_db1
        return dL_db1

    def dL_db2(self, y, y_pred):
        da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
        dz2_db2 = 1
        dL_da2 = dL_dy(y, y_pred)
        dL_db2 = dL_da2*da2_dz2*dz2_db2
        return dL_db2


class layer():
    def __init__(self, input_size, number_of_nuerons):
        self.input_size = input_size
        self.number_of_nuerons = number_of_nuerons
        self.w = np.random.random((input_size, number_of_nuerons))
        self.b = np.random.random((number_of_nuerons, 1))
        self.z = np.zeros((number_of_nuerons, 1), dtype=float)
        self.a = np.zeros((number_of_nuerons, 1), dtype=float)

    def set_zero(self):
        self.z = np.zeros((self.number_of_nuerons, 1), dtype=float)
        self.a = np.zeros((self.number_of_nuerons, 1), dtype=float)
        self.w = np.zeros(
            (self.input_size, self.number_of_nuerons), dtype=float)
        self.b = np.zeros((self.number_of_nuerons, 1), dtype=float)

    def calculate(self, X):
        self.z = np.dot(self.w.T, X) + self.b
        self.a = sigmoid(self.z)
        return self.a


def sigmoid(array):
    return 1/(1 + np.exp(-array))


def loss(y, y_pred):
    if y == 1:
        return -log(y_pred)
    else:
        return -log(1 - y_pred)


def dL_dy(y, y_pred):
    if y == 1:
        return -1/(y_pred)
    else:
        return 1/(1 - y_pred)


def sigmoid_dash(a):
    return sigmoid(a)*(1-sigmoid(a))


# y = np.array([[0],
#               [1],
#               [1],
#               [0]])
# y_pred = np.array([[0.001],
#                    [0.6],
#                    [0.6],
#                    [0.001]])
# cost = 0
# for j in range(y.shape[0]):
#     cost = cost + loss(y[j, 0], y_pred[j, 0])
#     cost = cost/y.shape[0]

# print(cost)
