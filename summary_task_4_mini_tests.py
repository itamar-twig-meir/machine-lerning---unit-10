import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from unit10 import c2w1_init_utils as u10
from DLLayer_and_DLModel_deep_network import *


#region data loading tests
plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()
#endregion

np.random.seed(1)

#region w initialization tests
hidden1 = DLLayer("Perseptrons 1", 30,(12288,),"relu",W_initialization = "Xavier",learning_rate = 0.0075, optimization='adaptive')
hidden2 = DLLayer("Perseptrons 2", 15,(30,),"trim_sigmoid",W_initialization = "He",learning_rate = 0.1)
#print(hidden1)
#print(hidden2)
#endregion

#region w and b save tests
hidden1 = DLLayer("Perseptrons 1", 10,(10,),"relu",W_initialization = "Xavier",learning_rate = 0.0075)
hidden1.b = np.random.rand(hidden1.b.shape[0], hidden1.b.shape[1])
hidden1.save_weights("SaveDir","Hidden1")
hidden2 = DLLayer ("Perseptrons 2", 10,(10,),"trim_sigmoid",W_initialization = "SaveDir/Hidden1.h5",learning_rate = 0.1)
#print(hidden1)
#print(hidden2)
m1 = DLModel()
m1.add_layer(hidden1)
m1.add_layer(hidden2)
dir = "m1"
m1.save_weights(dir)
print(os.listdir(dir))
#endregion

#region testing the 3 initialization types

#region zeros
"""
hidden_layer1 = DLLayer("Hidden1", 10, (2,) ,"relu","zeros", learning_rate= 0.01)
hidden_layer2 = DLLayer("Hidden2", 5, (10,) ,"relu","zeros", learning_rate= 0.01)
output_Layer = DLLayer("Output Layer", 1, (5,) ,"trim_sigmoid","zeros", learning_rate=0.1)

model = DLModel()
model.add_layer(hidden_layer1)
model.add_layer(hidden_layer2)
model.add_layer(output_Layer)
model.compile("cross_entropy")

costs = model.train(train_X, train_Y, 15000)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title("Learning rate =" + str(0.1))
plt.show()
"""
#endregion

#region random
"""
np.random.seed(1)
hidden_layer1 = DLLayer("Hidden1", 10, (2,) ,"relu","random", random_scale= 10, learning_rate= 0.01)
hidden_layer2 = DLLayer("Hidden2", 5, (10,) ,"relu","random", random_scale= 10, learning_rate= 0.01)
output_Layer = DLLayer("Output Layer", 1, (5,) ,"trim_sigmoid","random", random_scale= 10, learning_rate=0.1)

model = DLModel()
model.add_layer(hidden_layer1)
model.add_layer(hidden_layer2)
model.add_layer(output_Layer)
model.compile("cross_entropy")

init = "large random"
costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(init + " initialization")
plt.show()

plt.title("Model with " + init + " initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)

predictions = model.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')
predictions = model.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T))/float(test_Y.size)*100) + '%')
plt.show()
"""
#endregion

#region He
"""
np.random.seed(2)
hidden_layer1 = DLLayer("Hidden1", 10, (2,) ,"relu","he", learning_rate= 0.01)
hidden_layer2 = DLLayer("Hidden2", 5, (10,) ,"relu","he", learning_rate= 0.01)
output_Layer = DLLayer("Output Layer", 1, (5,) ,"trim_sigmoid","xavier", learning_rate=0.1)

model = DLModel()
model.add_layer(hidden_layer1)
model.add_layer(hidden_layer2)
model.add_layer(output_Layer)
model.compile("cross_entropy")

init = "he"
costs = model.train(train_X, train_Y, 15000)
plt.plot(costs)
plt.ylabel('cost')
plt.xlabel('iterations (per 150s)')
plt.title(init + " initialization")
plt.show()

plt.title("Model with " + init + " initialization")
axes = plt.gca()
axes.set_xlim([-1.5,1.5])
axes.set_ylim([-1.5,1.5])
u10.plot_decision_boundary(lambda x: model.predict(x.T), test_X, test_Y)

predictions = model.predict(train_X)
print ('Train accuracy: %d' % float((np.dot(train_Y,predictions.T) + np.dot(1-train_Y,1-predictions.T))/float(train_Y.size)*100) + '%')
predictions = model.predict(test_X)
print ('Test accuracy: %d' % float((np.dot(test_Y,predictions.T) + np.dot(1-test_Y,1-predictions.T))/float(test_Y.size)*100) + '%')
plt.show()
"""
#endregion

#endregion


