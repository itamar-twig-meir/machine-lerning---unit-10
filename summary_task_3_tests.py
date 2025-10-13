import numpy as np
import matplotlib.pyplot as plt
from DLLayer_and_DLModel import DLLayer, DLModel
import random
import sklearn
import sklearn.datasets
import sklearn.linear_model
from unit10 import c1w3_utils as u10


"""
# region inti + str + inti_weights test
np.random.seed(1)
l = [None]
l.append(DLLayer("Hidden 1", 6, (4000,)))
#print(l[1])
l.append(DLLayer("Hidden 2", 12, (6,),"leaky_relu", "random", 0.5,"adaptive"))
l[2].adaptive_cont = 1.2
#print(l[2])
l.append(DLLayer("Neurons 3",16, (12,),"tanh"))
#print(l[3])
l.append(DLLayer("Neurons 4",3, (16,),"sigmoid", "random", 0.2, "adaptive"))
l[4].random_scale = 10.0
l[4].init_weights("random")
#print(l[4])
# endregion

# region activation + trim tests
Z = np.array([[1,-2,3,-4],
              [-10,20,30,-40]])
l[2].leaky_relu_d = 0.1
# for i in range(1, len(l)):
    # print(l[i].activation_forward(Z))
# endregion

# region forward_propagation tests
np.random.seed(2)
m = 3
X = np.random.randn(4000,m)
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    # rint('layer',i," A", str(Al.shape), ":\n", Al)
# endregion

#region backward_activation tests
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, True)
    dZ = l[i].activation_backward(Al)
    print('layer',i," dZ", str(dZ.shape), ":\n", dZ)
#endregion

#region backwards_propagation tests
Al = X
for i in range(1, len(l)):
    Al = l[i].forward_propagation(Al, False)

np.random.seed(3)
fig, axes = plt.subplots(1, 4, figsize=(12,16))
fig.subplots_adjust(hspace=0.5, wspace=0.5)
dAl = np.random.randn(Al.shape[0],m) * np.random.random_integers(-100, 100, Al.shape)
for i in reversed(range(1,len(l))):
    axes[i-1].hist(dAl.reshape(-1), align='left')
    axes[i-1].set_title('dAl['+str(i)+']')
    dAl = l[i].backward_propagation(dAl)
#plt.show()

#endregion

# region learning + adaptive learning tests
np.random.seed(4)
random.seed(4)

l1 = DLLayer("Hidden1", 3, (4,),"trim_sigmoid", "zeros", 0.2, "adaptive")
l2 = DLLayer("Hidden2", 2, (3,),"relu", "random", 1.5)

#print("before update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
#print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))

l1.dW = np.random.randn(3,4) * random.randrange(-100,100)
l1.db = np.random.randn(3,1) * random.randrange(-100,100)
l2.dW = np.random.randn(2,3) * random.randrange(-100,100)
l2.db = np.random.randn(2,1) * random.randrange(-100,100)
l1.update_parameters()
l2.update_parameters()
#print("after update:W1\n"+str(l1.W)+"\nb1.T:\n"+str(l1.b.T))
#print("W2\n"+str(l2.W)+"\nb2.T:\n"+str(l2.b.T))

#endregion
""" #DLLayer tests

"""
#region loss function test
np.random.seed(1)
m1 = DLModel()
AL = np.random.rand(4,3)
Y = np.random.rand(4,3) > 0.7
m1.compile("cross_entropy")
errors = m1.loss_forward(AL,Y)
dAL = m1.loss_backward(AL,Y)
#print("cross entropy error:\n",errors)
#print("cross entropy dAL:\n",dAL)
m2 = DLModel()
m2.compile("squared_difference")
errors = m2.loss_forward(AL,Y)
dAL = m2.loss_backward(AL,Y)
#print("squared difference error:\n",errors)
#print("squared difference dAL:\n",dAL)
#endregion

#region compute cost test
#print("cost m1:", m1.compute_cost(AL,Y))
#("cost m2:", m2.compute_cost(AL,Y))
#endregion

#region train and predict tests
np.random.seed(1)
model = DLModel();
model.add_layer(DLLayer("Perseptrons 1", 10,(12288,)))
model.add_layer(DLLayer("Perseptrons 2", 1,(10,),"trim_sigmoid"))
model.compile("cross_entropy", 0.7)
X = np.random.randn(12288,10) * 256
#print("predict:",model.predict(X))
#endregion

#final test 
#print(model)
""" #DLModel test




np.random.seed(1)

# region idk, shape stuff
#plt.figure(1)
X, Y = u10.load_planar_dataset()
#plt.scatter(X[0, :], X[1, :], c=Y[0, :], s=40, cmap=plt.cm.Spectral);
#plt.show()

m = Y.shape[1]
X_shape = X.shape
Y_shape = Y.shape

#print ('The shape of X is: ' + str(X_shape))
#print ('The shape of Y is: ' + str(Y_shape))
#print ('I have m = %d training examples!' % (m))
#endregion

#region Train the logistic regression classifier
"""clf = sklearn.linear_model.LogisticRegressionCV();
clf.fit(X.T, Y[0,:])
# Plot the decision boundary for logistic regression
u10.plot_decision_boundary(lambda x: clf.predict(x), X, Y)
plt.title("Logistic Regression")
plt.show()
# Print accuracy
LR_predictions = clf.predict(X.T)
print ('Accuracy of logistic regression: %d ' % float((np.dot(Y,LR_predictions) + np.dot(1-Y,1-LR_predictions))/float(Y.size)*100) +
      '% ' + "(percentage of correctly labelled datapoints)")"""
#endregion

#region Make the neural network

hidden_layer = DLLayer("Hidden1", 4, (2,) ,"tanh", learning_rate= 0.1)
output_Layer = DLLayer("Output Layer", 1, (4,) ,"sigmoid", learning_rate=0.5)



model = DLModel();
model.add_layer(hidden_layer)
model.add_layer(output_Layer)
model.compile("cross_entropy")

#print(model)
#endregion

#region Train the model
costs = model.train(X,Y,10000)
"""plt.figure(2)
plt.plot(np.squeeze(costs))
plt.ylabel('cost')
plt.show()"""
#endregion

#region final test
u10.plot_decision_boundary(lambda x: model.predict(x.T), X, Y)
plt.title("Decision Boundary for hidden layer size " + str(4))
plt.show()
predictions = model.predict(X)
print ('Accuracy: %d' % float((np.dot(Y,predictions.T) + np.dot(1-Y,1-predictions.T))/float(Y.size)*100) + '%')
#endregion