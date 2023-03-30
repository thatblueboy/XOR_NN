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
        # print('predicted')
        # print(y_predicted)
        return float(y_predicted[0])

    # def gradient(x, w1, a1, b1, z1, w2, a2, b2, z2, y):
    #     da2_dz2 = sigmoid_dash(z2)
    #     dz2_db2 = b2
    #     dL_da2 = dL_dy(y, a2)

    #     dL_db2 = dL_da2*da2_dz2*dz2_db2

        # dz2/dw2[i, 0] = a1[i, 0]
        # dL/dw2[i, 0] = a1[i, 0] * dL_da2 * da2_dz2

        # dz2/da1[i, 0] = w2[i, 0]
        # da1[i, 0]/db1[i, 0] = b1[i, 0]

        # dz2/da1[j, 0] = a1[j, 0]
        # da1[j, 0]/dw[i, j] = X[1, 0]

    def gradient_descent(self, a, epochs, x,  y):
        for i in range(epochs):
            y_pred = np.ones((y.shape[0], y.shape[1]), dtype= float)
            for k in range(y_pred.shape[0]):
                y_pred[k, 0] = self.calculate(x[k:k+1].T)
            print(y)
            print(y_pred)
            self.update(a, x, y, y_pred)
            self.layer_1.b = self.layer_1_update.b
            self.layer_1.w = self.layer_1_update.w
            self.layer_2.b = self.layer_2_update.b
            self.layer_2.w = self.layer_2_update.w

            cost = 0
            for j in range(y.shape[0]):
                cost = cost + loss(y[j, 0], y_pred[j, 0])
            cost = cost/y.shape[0]
            print('cost after epoch %i is %f' %(i, cost))

    def update(self, a, x, y, y_pred):
        
        

        self.layer_2_update.b[0, 0] = self.layer_2.b[0,
                                                     0] - a * self.gradientb2(y, y_pred)

        for i in range(self.layer_2.w.shape[0]):
            self.layer_2_update.w[i, 0] = self.layer_2.w[i,
                                                         0] - a * self.gradientw2(y, y_pred, i)

        for i in range(self.layer_1.b.shape[0]):
            self.layer_1_update.b[i, 0] = self.layer_1.b[i,
                                                         0] - a * self.gradientb1(y, y_pred, i)
        for i in range(self.layer_1.w.shape[1]):
            for j in range(self.layer_1.w.shape[0]):
                self.layer_1_update.w[j, i] = self.layer_1.w[j,
                                                             i] - a * self.gradientw1(x, y, y_pred, i, j)

    def gradientw1(self, x, y, y_pred,  i, j):
        gradient = 0
        num = y.shape[0]
        for k in range(num):
            dL_da2 = dL_dy(y[k, 0], y_pred[k, 0])
            da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
            dz2_da1 = self.layer_2.w[i, 0]  # a1 = layer1.a[j, 0]
            da1_dw1 = x[k, j]
            gradient = dL_da2 * da2_dz2 * dz2_da1 * da1_dw1
        gradient = gradient/num
        return gradient

    def gradientw2(self, y, y_pred,  j):
        gradient = 0
        num = y.shape[0]
        for i in range(num):
            dL_da2 = dL_dy(y[i, 0], y_pred[i, 0])
            da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
            dz2_dw2 = self.layer_1.a[j, 0]  # w2 = layer2.w[j, 0]
            gradient = dL_da2 * da2_dz2 * dz2_dw2
        gradient = gradient/num
        return gradient

    def gradientb1(self, y, y_pred,  j):
        gradient = 0
        num = y.shape[0]
        for i in range(num):
            dL_da2 = dL_dy(y[i, 0], y_pred[i, 0])
            da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
            dz2_da1 = self.layer_2.w[j, 0]  # a1 = layer1.a[j, 0]
            da1_db1 = self.layer_1.b[j, 0]
            gradient = dL_da2 * da2_dz2 * dz2_da1 * da1_db1
        gradient = gradient/num
        return gradient

    def gradientb2(self, y, y_pred):
        gradient = 0
        num = y.shape[0]
        for i in range(num):
            da2_dz2 = sigmoid_dash(self.layer_2.z[0, 0])
            dz2_db2 = 1
            dL_da2 = dL_dy(y[i, 0], y_pred[i, 0])
            gradient = gradient + dL_da2*da2_dz2*dz2_db2
        gradient = gradient/num
        return gradient


class layer():
    def __init__(self, input_size, number_of_nuerons):
        self.w = np.random.random((input_size, number_of_nuerons))
        self.b = np.random.random((number_of_nuerons, 1))
        self.z = np.zeros((number_of_nuerons, 1), dtype=float)
        self.a = np.zeros((number_of_nuerons, 1), dtype=float)

    def calculate(self, X):
        # print(self.w.T)
        # print(X)
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
        # print('1')
        return -1/(y_pred)
    else:
        # print('0')

        return 1/(1 - y_pred)


def sigmoid_dash(a):
    return sigmoid(a)*(1-sigmoid(a))


y = np.array([[0],
              [1],
              [1],
              [0]])
y_pred = np.array([[0.001],
              [0.6],
              [0.6],
              [0.001]])
cost = 0
for j in range(y.shape[0]):
    cost = cost + loss(y[j, 0], y_pred[j, 0])
    cost = cost/y.shape[0]

print(cost)