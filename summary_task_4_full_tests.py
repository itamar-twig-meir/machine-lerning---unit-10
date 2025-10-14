import time
import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from PIL import Image
from scipy import ndimage
from unit10 import c1w4_utils as u10
from DLLayer_and_DLModel_deep_network import *


plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
np.random.seed(1)


train_x, train_y, test_x, test_y, classes = u10.load_datasetC1W4()
#region Example of a picture
"""
index = 2
plt.imshow(train_x[index])
plt.show()
print ("y = " + str(train_y[0,index]) + ". It's a " + classes[train_y[0,index]].decode("utf-8") +  " picture.")
"""
#endregion

#region shape stuff
num_px = train_x[0].shape[0]
m_train = train_x.shape[0]
m_test = test_x.shape[0]
num_rgb = np.prod(train_x[0].shape)

"""
print ("Number of training examples: " + str(m_train))
print ("Number of testing examples: " + str(m_test))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_x shape: " + str(train_x.shape))
print ("train_y shape: " + str(train_y.shape))
print ("test_x shape: " + str(test_x.shape))
print ("test_y shape: " + str(test_y.shape))
"""
#endregion

#region flattening and normalization
train_x_flatten = train_x.reshape(train_x.shape[0], - 1).T
test_x_flatten = test_x.reshape(test_x.shape[0], - 1).T
train_x = (train_x_flatten /255) - 0.5
test_x = (test_x_flatten /255) - 0.5
"""
print ("train_x's shape: " + str(train_x.shape))
print ("test_x's shape: " + str(test_x.shape))
print ("normelized train color: ", str(train_x[10][10]))
print ("normelized test color: ", str(test_x[10][10]))
"""
#endregion

#region shallow network
"""
hidden_layer = DLLayer("hidden_layer",7 ,(num_rgb,),"relu","xavier", 0.007)
output_Layer = DLLayer("output_layer",1 ,(7,),"sigmoid","xavier", 0.007)
model = DLModel()
model.add_layer([hidden_layer,output_Layer])
model.compile("cross_entropy")

costs = model.train(train_x, train_y,2500)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_x) == train_y))
print("test accuracy:", np.mean(model.predict(test_x) == test_y))
"""
#endregion

#region deep network
"""
hidden_layer1 = DLLayer("hidden_layer1",30 ,(num_rgb,),"relu","xavier", 0.0075)
hidden_layer2 = DLLayer("hidden_layer2",15 ,(30,),"relu","xavier", 0.0075)
hidden_layer3 = DLLayer("hidden_layer3",10 ,(15,),"relu","xavier", 0.0075)
hidden_layer4 = DLLayer("hidden_layer4",10 ,(10,),"relu","xavier", 0.0075)
hidden_layer5 = DLLayer("hidden_layer5",5 ,(10,),"relu","xavier", 0.0075)
output_Layer = DLLayer("output_layer",1 ,(5,),"sigmoid","xavier", 0.0075)
model = DLModel()
model.add_layer([hidden_layer1, hidden_layer2, hidden_layer3, hidden_layer4, hidden_layer5, output_Layer])
model.compile("cross_entropy")

costs = model.train(train_x, train_y,2500)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 25s)')
plt.title("Learning rate =" + str(0.007))
plt.show()
print("train accuracy:", np.mean(model.predict(train_x) == train_y))
print("test accuracy:", np.mean(model.predict(test_x) == test_y))
"""
#endregion
















