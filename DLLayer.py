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
            self._adaptive_alpha_b = np.full((self.num_units, 1), self.learning_rate)
            self._adaptive_alpha_W = np.full((self.num_units, *(self.input_shape)), self.learning_rate)
        self.init_weights(self, W_initialization)

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
        print ("hi")
        s += "\tparameters:\n\t\tb.T: " + str(self.b.T) + "\n"
        s += "\t\tshape weights: " + str(self.W.shape) + "\n"
        plt.hist(self.W.reshape(-1))
        plt.title("W histogram")
        plt.show()
        return s

    def  activation_forward(self, Z):
        match self.activation:
            case "sigmoid":
                return self.sigmoid(Z)
            case "trim_sigmoid":
                return self._trim_sigmoid(Z)
            case "tanh":
                 return self._tanh(Z)
            case "trim_tanh":
                return self._trim_tanh(Z)
            case "relu":
                return self._relu(Z)
            case "leaky_relu":
                 return self._leaky_relu(Z)



    def sigmoid(self, Z):
        return 1/1 + np.exp(-Z)

    def _trim_sigmoid(self, Z):
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

    def _tanh(self, Z):
        return np.tanh(Z)

    def _trim_tanh(self, Z):
        A = np.tanh(Z)
        TRIM = self.activation_trim
        if TRIM > 0:
            A = np.where(A < -1 + TRIM, TRIM, A)
            A = np.where(A > 1 - TRIM, 1 - TRIM, A)
        return A

    def _relu(self, Z):
        return np.maximum(0, Z, )

    def _leaky_relu(self, Z):
        return np.where(Z > 0, Z, Z * self.leaky_relu_d)


    #def forward_propagation():
    #def backward_propagation():
    #def update_parameters():

