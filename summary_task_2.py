import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from numpy.ma.core import reshape
from scipy import ndimage
from PIL import Image
from unit10 import c1w2_utils as u10


train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()

m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig[0].shape[0]
num_rgb = np.prod(train_set_x_orig[0].shape)


"""""
index = 0
plt.imshow(train_set_x_orig[index])
plt.show()
print ("y = " + str(train_set_y[index]) + ", it's a '" +
classes[np.squeeze(train_set_y[index])].decode("utf-8") +  "' picture.")
""" # to see a picture


"""""
print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
""" # tests + shape checking


# region flattening the pictures + normalization

# flattening the arrays so they contain a list of pictures as a long list of rgb values.
flattened_train_set_x = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
flattened_test_set_x = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

flattened_train_set_x = np.array(flattened_train_set_x)
flattened_test_set_x = np.array(flattened_test_set_x)

final_train_set_x = flattened_train_set_x/255.0
final_test_set_x = flattened_test_set_x/255.0

final_train_set_y = np.array(train_set_y)
final_test_set_y = np.array(test_set_y)
final_train_set_y = final_train_set_y.reshape(1,final_train_set_y.shape[0])
final_test_set_y = final_test_set_y.reshape(1,final_test_set_y.shape[0])

# tests
"""
for i in range(3):
    print(str(flattened_train_set_x[i][0]) + " = " + str(train_set_x_orig[0][0][0][i]))
print ("train_set_x_flatten shape: " + str( flattened_train_set_x.shape))
print ("train_set_y shape: " + str(final_train_set_y.shape))
print ("test_set_x_flatten shape: " + str( flattened_test_set_x.shape))
print ("test_set_y shape: " + str(final_test_set_y.shape))
"""

# endregion


#region function def

def sigmoid (z):
    return 1/(1+np.exp(-z))

# sigmoid test - print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    return w.T, 0

""" 
W, b = initialize_with_zeros(2)
print ("W = " + str(W))
print ("b = " + str(b))
""" # initializing tests

def forward_propagation(X, Y, w, b):

    y_pred = sigmoid((np.dot(np.transpose(w), X) + b))
    mistake_avr = np.sum( -(Y* np.log(y_pred) + (1 - Y)* np.log(1 - y_pred)))

    return y_pred, mistake_avr / X.shape[1]

w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1])
A, cost = forward_propagation(X, Y, w, b)
print ("cost = " + str(cost))


#endregion

