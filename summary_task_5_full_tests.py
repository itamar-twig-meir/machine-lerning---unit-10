from sklearn.datasets import fetch_openml
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image, ImageOps

from DLLayer_and_DLModel_softmax import *

mnist = fetch_openml('mnist_784')
X, Y = mnist["data"], mnist["target"]
X=np.array(X)
Y=np.array(Y)

X = X / 255 - 0.5

"""i = 12
plt.imshow(X[i:i+1].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print("Label is: '"+Y[i]+"'")
"""

Y_new = DLModel.to_one_hot(10,Y)
m = 60000
m_test = X.shape[0] - m
X_train, X_test = X[:m].T, X[m:].T
Y_train, Y_test = Y_new[:,:m], Y_new[:,m:]
np.random.seed(111)
shuffle_index = np.random.permutation(m)
X_train, Y_train = X_train[:, shuffle_index], Y_train[:, shuffle_index]
"""i = 12
plt.imshow(X_train[:,i].reshape(28,28), cmap = matplotlib.cm.binary)
plt.axis("off")
plt.show()
print(Y_train[:,i])"""


layer1 = DLLayer(64,(784,),"layer1","sigmoid", "xavier",0.1, "adaptive" )
layer2 = DLLayer(10,(64,),  "layer2", "softmax", "xavier",0.1, "adaptive" )
model = DLModel()
model.add_layer(layer1)
model.add_layer(layer2)

model.compile("categorical_cross_entropy")

print("starting")
np.random.seed(1)
costs = model.train(X_train, Y_train, 200)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations')
plt.title("Learning rate =" + str(0.1))
plt.show()
model.save_weights("digits")

model.confusion_matrix(X_train, Y_train)
print("Test:")
model.confusion_matrix(X_test, Y_test)
