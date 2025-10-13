from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from numpy.ma.core import reshape
from scipy import ndimage
from PIL import Image
from unit10 import c1w2_utils as u10


class DLLayer:

    def __init__(self, name, num_units, input_shape, activation = "relu", W_initialization = "random", learning_rate = 0.001, optimization = "none", random_scale = 0.01 ,leaky_relu_d = 0.001):
        self.name = name
        self.num_units = num_units
        self.input_shape = input_shape
        self.activation = activation
        self.W_initialization = W_initialization
        self.learning_rate = learning_rate
        self.optimization = optimization
        self.random_scale = random_scale
        self.leaky_relu_d = leaky_relu_d
        self.activation_trim = 1e-10
        if optimization == "adaptive" :
            self.adaptive_alpha_b = np.full((self.num_units, 1), self.learning_rate)
            self.adaptive_alpha_W = np.full((self.num_units, *(self.input_shape)), self.learning_rate)
            self.adaptive_cont = 1.2
            self.adaptive_switch = 0.5
        self.init_weights(W_initialization)

    def init_weights(self, W_initialization):
        self.b = np.zeros((self.num_units, 1), dtype=float)

        if(self.W_initialization == "random"):
            self.W = np.random.randn(self.num_units, *(self.input_shape)) * self.random_scale
        else:
            self.W = np.zeros((self.num_units, *self.input_shape), dtype=float)

    def __str__(self):
        s = self.name + " Layer:\n"
        s += "\tnum_units: " + str(self.num_units) + "\n"
        s += "\tactivation: " + self.activation + "\n"
        if self.activation == "leaky_relu":
            s += "\t\tleaky relu parameters:\n"
           # s += "\t\t\tleaky_relu_d: " + str(self.leaky_relu_d) + "\n"
        s += "\tinput_shape: " + str(self.input_shape) + "\n"
        s += "\tlearning_rate (alpha): " + str(self.learning_rate) + "\n"
        # optimization
        if self.optimization == "adaptive":
            s += "\t\tadaptive parameters:\n"
            #s += "\t\t\tcont: " + str(self.adaptive_cont) + "\n"
            #s += "\t\t\tswitch: " + str(self.adaptive_switch) + "\n"
        # parameters

        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape) + "\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s

    def forward_propagation(self, A_prev, is_predict):
        self.A_prev = np.array(A_prev, copy=True)
        self.Z = np.dot( self.W , self.A_prev ) + self.b

        return self.activation_forward(self.Z)

    # region forward_activation
    def  activation_forward(self, Z):
        match self.activation:
            case "sigmoid":
                return self.sigmoid(Z)
            case "trim_sigmoid":
                return self.trim_sigmoid(Z)
            case "tanh":
                 return self.tanh(Z)
            case "trim_tanh":
                return self.trim_tanh(Z)
            case "relu":
                return self.relu(Z)
            case "leaky_relu":
                 return self.leaky_relu(Z)

            case _:
                print("no valid activation - error? \n relu has been chosen ")
                self.activation = "relu"
                return self.relu(Z)

    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))

    def trim_sigmoid(self, Z):
        with np.errstate(over='raise', divide='raise'):
            try:
                A = 1 / (1 + np.exp(-Z))
            except FloatingPointError:
                Z = np.where(Z < -100, -100, Z)
                A = A = 1 / (1 + np.exp(-Z))
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def tanh(self, Z):
        return np.tanh(Z)

    def trim_tanh(self, Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def relu(self, Z):
        return np.maximum(0, Z, )

    def leaky_relu(self, Z):
        return np.where(Z > 0, Z, Z * self.leaky_relu_d)
    #endregion

    def backward_propagation(self, dA):
        dZ = self.activation_backward(dA)
        self.dW = 1/self.A_prev.shape[1] * np.dot(dZ, self.A_prev.T)
        self.db = 1/self.A_prev.shape[1] * np.sum(dZ, axis = 1, keepdims = True)
        return np.dot(self.W.T, dZ)

    # region backward_activation
    def activation_backward(self, dA):
        match self.activation:
            case "sigmoid" | "trim_sigmoid":
                return self.sigmoid_backward(dA)
            case "tanh" | "trim_tanh":
                return self.tanh_backward(dA)
            case "relu":
                return self.relu_backward(dA)
            case "leaky_relu":
                return self.leaky_relu_backward(dA)
            case _:
                print("no valid backward activation, error? \n relu has been chosen ")
                return self.relu_backward(dA)

    def sigmoid_backward(self, dA):
        A = self.sigmoid(self.Z)
        dZ = dA * A * (1 - A)
        return dZ

    def tanh_backward(self, dA):
        A = self.tanh(self.Z)
        dZ = dA * (1 - A**2)
        return dZ

    def relu_backward(self, dA):
        return np.where(self.Z <= 0, 0, dA)

    def leaky_relu_backward(self, dA):
        return np.where(self.Z <= 0, dA * self.leaky_relu_d, dA)

    #endregion

    def update_parameters(self):
        if self.optimization == "none":
            self.W -= self.dW * self.learning_rate
            self.b -= self.db * self.learning_rate
        else:

            same_sign_W = np.sign(self.dW) == np.sign(self.adaptive_alpha_W)
            same_sign_b = np.sign(self.db) == np.sign(self.adaptive_alpha_b)

            self.adaptive_alpha_W *= np.where(same_sign_W, self.adaptive_cont, -self.adaptive_switch)
            self.adaptive_alpha_b *= np.where(same_sign_b, self.adaptive_cont, -self.adaptive_switch)

            self.W -= self.adaptive_alpha_W
            self.b -= self.adaptive_alpha_b




class DLModel:

    def __init__(self, name = "Model" ):
        self.name = name
        self.is_compiled = False
        self.layers = [None]

    def add_layer(self, layer):
        if(isinstance(layer ,DLLayer)):
            self.layers.append(layer)
        else:
            print("error, tried adding a non layer object to layers array")

    def squared_difference(self, AL, Y):

        return ((AL - Y) ** 2)

    def squared_difference_backward(self, AL, Y):

        return 2*(AL - Y)

    def cross_entropy(self, AL, Y):

        return np.where(Y == 0, -np.log(1 - AL), -np.log(AL))

    def cross_entropy_backward(self, AL, Y):

        return np.where(Y == 0, 1/(1-AL), -1/AL)

    def loss_func(self, loss):
        match loss:
            case "squared_difference":
                return self.squared_difference, self.squared_difference_backward
            case "cross_entropy":
                return self.cross_entropy, self.cross_entropy_backward

    def compile(self, loss, threshold=0.5):
        self.loss = loss
        self.threshold = threshold
        self.is_compiled = True
        self.loss_forward, self.loss_backward = self.loss_func(loss)

    def compute_cost(self, AL, Y):
        m = Y.shape[1]
        return np.sum(self.loss_forward(AL, Y)) / m

    def train(self, X, Y, num_iterations):
        print_ind = max(num_iterations // 100, 1)
        L = len(self.layers)
        costs = []
        for i in range(num_iterations):
            # forward propagation
            Al = X
            for l in range(1, L):
                Al = self.layers[l].forward_propagation(Al, False)
                # backward propagation
            dAl = self.loss_backward(Al, Y)
            for l in reversed(range(1, L)):
                dAl = self.layers[l].backward_propagation(dAl)
                # update parameters
                self.layers[l].update_parameters()
            # record progress
            if i > 0 and i % print_ind == 0:
                J = self.compute_cost(Al, Y)
                costs.append(J)
                print("cost after ", str(i // print_ind), "%:", str(J))
        return costs

    def predict(self, X):
        AL = X
        for layer in self.layers[1:]:
            AL = layer.forward_propagation(AL, is_predict=True)
        return np.where(AL > self.threshold, True, False)

    def __str__(self):
        s = self.name + " description:\n\tnum_layers: " + str(len(self.layers) - 1) + "\n"
        if self.is_compiled:
            s += "\tCompilation parameters:\n"
            s += "\t\tprediction threshold: " + str(self.threshold) + "\n"
            s += "\t\tloss function: " + self.loss + "\n\n"

        for i in range(1, len(self.layers)):
            s += "\tLayer " + str(i) + ":" + str(self.layers[i]) + "\n"
        return s




