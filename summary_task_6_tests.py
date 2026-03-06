from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import os
import h5py
import scipy
from numpy.ma.core import reshape
from scipy import ndimage
from PIL import Image
from sklearn.metrics import classification_report, confusion_matrix
from DLLayer_and_DLModel_mini_batch import *
import time
from unit10 import c2w2_utils #importent!! this is one of the changes that fixes the problem with the presentation


plt.rcParams['figure.figsize'] = (7.0, 4.0) # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

X_assess, Y_assess, mini_batch_size = c2w2_utils.random_mini_batches_test_case()#importent!! this is one of the changes that fixes the problem with the presentation
mini_batches = DLModel.random_mini_batches(X_assess, Y_assess, mini_batch_size, seed = 0)
print ("shape of the 1st mini_batch_X: " + str(mini_batches[0][0].shape))
print ("shape of the 2nd mini_batch_X: " + str(mini_batches[1][0].shape))
print ("shape of the 3rd mini_batch_X: " + str(mini_batches[2][0].shape))
print ("shape of the 1st mini_batch_Y: " + str(mini_batches[0][1].shape))
print ("shape of the 2nd mini_batch_Y: " + str(mini_batches[1][1].shape))
print ("shape of the 3rd mini_batch_Y: " + str(mini_batches[2][1].shape))
print ("mini batch sanity check: " + str(mini_batches[0][0][0][0:3]))

train_X, train_Y = c2w2_utils.load_dataset()#importent!! this is one of the changes that fixes the problem with the presentation
plt.show()
layer1 = DLLayer(64,(train_X.shape[0],), name='layer1', activation = 'relu', W_initialization="he", learning_rate=0.05, optimization= "none")
layer2 = DLLayer(32,(64,), name='layer2', activation = 'relu', W_initialization="he", learning_rate=0.05, optimization= "none")
layer3 = DLLayer(5,(32,), name='layer3', activation = 'relu', W_initialization="he", learning_rate=0.05, optimization= "none")
layer4 = DLLayer(1,(5,), name='layer4', activation = 'trim_sigmoid', W_initialization="he", learning_rate=0.05, optimization= "none")
model = DLModel("model")

model.add_layer(layer1)
model.add_layer(layer2)
model.add_layer(layer3)
model.add_layer(layer4)
model.compile(loss="cross_entropy", threshold=0.5)


def run_model(model, num_epocs, minibatch_size):
    tic = time.time()
    costs = model.train(train_X, train_Y, num_epocs, minibatch_size)
    toc = time.time()
    print (f"time (ms): {1000*(toc-tic)}")

    c2w2_utils.print_costs(costs,num_epocs)#importent!! this is one of the changes that fixes the problem with the presentation
    train_predict = model.forward_propagation(train_X) > 0.7
    accuracy = np.sum(train_predict == train_Y)/train_X.shape[1]
    print("accuracy:", str(accuracy))
    #plt.title("Model with no mini batches")
    axes = plt.gca()
    axes.set_xlim([-1.5,2.5])
    axes.set_ylim([-1,1.5])
    c2w2_utils.plot_decision_boundary(model, train_X, train_Y)#importent!! this is one of the changes that fixes the problem with the presentation

run_model(model, 4000,64)