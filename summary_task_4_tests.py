import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from unit10 import c2w1_init_utils as u10
from DLLayer_and_DLModel import *

plt.rcParams['figure.figsize'] = (7.0, 4.0)
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'

# load image dataset: blue/red dots in circles
train_X, train_Y, test_X, test_Y = u10.load_dataset()
plt.show()