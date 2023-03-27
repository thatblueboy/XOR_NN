import numpy as np
import matplotlib
from math import log


x = np.array([[0, 0],
              [0, 1],
              [1, 0],
              [1, 1]])
print(x.shape)

y = np.array([[0],
              [1],
              [1],
              [0]])


def cost(y, y_pred):

    one = np.ones((y.shape[0], y.shape[1]), dtype=float)
    # print(one)
    # # print(log(1-y_pred))
    # print(y)
    # print(np.log(one-y_pred))
    # print((one-y))
    # print(np.log(one-y_pred))
    cost = -np.dot(y.T, np.log(one-y_pred)) - \
        np.dot((one-y).T, np.log(one-y_pred))
    cost = cost/y.shape[0]
    return float(cost)


def sigmoid(array):
    return 1/(1 + np.exp(-array))


class layer():
    def __init__(self, input_size, number_of_nuerons):
        self.inp_size = input_size
        self.units = number_of_nuerons
        self.w = np.zeros((number_of_nuerons, input_size), dtype=float)
        self.b = np.zeros((number_of_nuerons, ), dtype=float)

    def calculate(self, input):
        # num_features = input.shape[1]
        num_datapoints = input.shape[0]
        print(num_datapoints)

        activation = np.zeros((num_datapoints, self.units), dtype=float)

        for i in range(num_datapoints):

            for j in range(self.units):

                activation[i][j] = np.dot(self.w[j], input[i]) + self.b[j]
                # print(activation[i][0])
                pass
        activation = sigmoid(activation)
        print(activation)
        return activation
        # print(self.w)


arr = np.array([[0, 3],
                [5, 0]])

# print(sigmoid(arr))
column = np.zeros((2, 1), dtype=float)
print(arr[1])
column = arr[:, 1]
print(column)
layer_1 = layer(2, 16)
layer_1.w = np.array([[1, 2],
                      [4, 5],
                      [7, 8],
                      [1, 2],
                      [4, 5],
                      [7, 8],
                      [1, 2],
                      [4, 5],
                      [7, 8],
                      [1, 2],
                      [4, 5],
                      [7, 8],
                      [1, 2],
                      [4, 5],
                      [7, 8],
                      [1, 2]])
layer_1.b = np.array([1, 1, 1, 8, 1, 1, 1, 1, 1, 1, 1,
                      1, 1, 1, 1, 1])

layer_2 = layer(16, 1)
# layer_2.w = np.array([[1, 2, 3, 3, 4, 5, 6, 7, 7, 0, 8, 8, 3, 8, 8, 8]])
layer_2.w = np.zeros((1, 16), dtype=float)
layer_2.b = np.array([0])

# input has to be an numpy array with 2 dimensions, 1 dimensional array will throw an error
y_predicted = layer_2.calculate(layer_1.calculate(x))
print(cost(y, y_predicted))


# print(y_predicted)
# print(la)
