import numpy as np
import h5py
import matplotlib.pyplot as plt
from DLLayer_and_DLModel_softmax import *




# set default size of plots
plt.rcParams['figure.figsize'] = (5.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'
# set seed
np.random.seed(1)


np.random.seed(1)
softmax_layer = DLLayer (name = "Softmax 1", num_units= 3, input_shape = (4,), activation= "softmax", W_initialization= "random")
A_prev = np.random.randn(4, 5)
A = softmax_layer.forward_propagation(A_prev, False)
dA = A
dA_prev = softmax_layer.backward_propagation(dA)
dW = softmax_layer.dW
db = softmax_layer.db
#there was a problem in the tests here, this is a correction
"""
print("A:\n",A)
print("dA_prev:\n",dA_prev)
"""


np.random.seed(2)
softmax_layer = DLLayer(num_units=3, input_shape = (4,), activation="softmax" , W_initialization="random", learning_rate= 1.0, optimization='adaptive')
"""print("W before:\n",softmax_layer.W)
print("b before:\n",softmax_layer.b)"""
model = DLModel()
model.add_layer(softmax_layer)
model.compile("categorical_cross_entropy")
X = np.random.randn(4, 5)
Y = np.random.rand(3, 5)
Y = np.where(Y==Y.max(axis=0),1,0)
cost = model.train(X,Y,1)
"""
print("cost:",cost[0])
print("W after:\n",softmax_layer.W)
print("b after:\n",softmax_layer.b)"""
#idk why there is a different after values


np.random.seed(3)
softmax_layer = DLLayer( name = "Layer1",num_units= 3 ,input_shape = (4,), activation= "trim_softmax", optimization='adaptive', W_initialization= "Xavier")
model = DLModel()
model.add_layer(softmax_layer)
model.compile("categorical_cross_entropy")
X = np.random.randn(4, 50000)*10
Y = np.zeros((3, 50000))
sumX = np.sum(X,axis=0)
for i in range (len(Y[0])):
    if sumX[i] > 5:
        Y[0][i] = 1
    elif sumX[i] < -5:
        Y[2][i] = 1
    else:
        Y[1][i] = 1
costs = model.train(X,Y,1000)
plt.plot(costs)
plt.show()
predictions = model.predict(X)
print("right",np.sum(Y.argmax(axis=0) == predictions.argmax(axis=0)))
print("wrong",np.sum(Y.argmax(axis=0) != predictions.argmax(axis=0)))















