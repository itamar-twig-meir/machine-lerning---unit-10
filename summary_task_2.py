import numpy as np
from unit10 import c1w2_utils as u10


#region uploading and initializing data
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = u10.load_datasetC1W2()
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig[0].shape[0]
num_rgb = np.prod(train_set_x_orig[0].shape)
#endregion

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


"""
for i in range(3):
    print(str(flattened_train_set_x[i][0]) + " = " + str(train_set_x_orig[0][0][0][i]))
print ("train_set_x_flatten shape: " + str( flattened_train_set_x.shape))
print ("train_set_y shape: " + str(final_train_set_y.shape))
print ("test_set_x_flatten shape: " + str( flattened_test_set_x.shape))
print ("test_set_y shape: " + str(final_test_set_y.shape))
""" # tests

# endregion


#region function def

def sigmoid (z):

    return 1/(1+np.exp(-z))

# sigmoid test - print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0,2]))))

def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    return w, 0

""" 
W, b = initialize_with_zeros(2)
print ("W = " + str(W))
print ("b = " + str(b))
""" # initializing tests

def forward_propagation(X, Y, w, b):

    y_pred = sigmoid(np.dot(w.T, X) + b)
    mistake_avr = np.sum( - (Y* np.log(y_pred) + (1 - Y)* np.log(1 - y_pred)))

    return y_pred, mistake_avr / X.shape[1]

""" 
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1])
A, cost = forward_propagation(X, Y, w, b)
print ("cost = " + str(cost))
""" # forward_propagation tests

def backward_propagation(X, Y, A):
    m = X.shape[1]
    dw = (1 / m) * np.dot( X , (A - Y).T)
    db = (1 / m) * np.sum(A - Y)

    return dw, db

"""
w, b, X, Y = np.array([[1.],[2.]]), 2., np.array([[1.,2.,-1.], [3.,4.,-3.2]]), np.array([1,0,1])
A, cost = forward_propagation(X, Y, w, b)
dw, db = backward_propagation(X, Y, A)
print ("dW = " + str(dw))
print ("db = " + str(db))
""" # backward_propagation tests

def train(X, Y, num_iterations, learning_rate, is_adaptive = False):
    w, b = initialize_with_zeros(X.shape[0])

    learning_rate_b = 1
    learning_rate_w = 1
    old_dw = np.zeros_like(w)
    old_db = 1
    if is_adaptive:
        learning_rate_b = 0.001
        learning_rate_w = np.full_like(w, 0.001)
        learning_rate = 1

    for i in range(num_iterations):
        A, J = forward_propagation(X,Y, w, b)
        dw, db = backward_propagation(X,Y, A)

        if is_adaptive:
            learning_rate_w, learning_rate_b = adaptive_learning(dw, old_dw, db, old_db, learning_rate_b, learning_rate_w)
        w -= dw * learning_rate * learning_rate_w
        b -= db * learning_rate * learning_rate_b
        old_dw = dw
        old_db = db
    return w,b

"""
X, Y = np.array([[1.,2.,-1.],[3.,4.,-3.2]]), np.array([1,0,1])
W, b = train(X, Y, num_iterations= 100, learning_rate = 0.009)
print ("W = " + str(W))
print ("b = " + str(b))
""" # train tests

def predict(X, W, b):
    y_pred = sigmoid(np.dot(W.T, X) + b)

    return (y_pred > 0.5).astype(int)

""" 
W = np.array([[0.1124579],[0.23106775]])
b = -0.3
X = np.array([[1.,-1.1,-3.2],[1.2,2.,0.1]])
print ("predictions = " + str(predict(X, W, b)))
"""# predict test

def adaptive_learning (dw, old_dw, db, old_db, learning_rate_b, learning_rate_w):

    sign_product = np.sign(dw) * np.sign(old_dw)
    learning_rate_w[sign_product > 0] *= 1.1
    learning_rate_w[sign_product < 0] *= 0.5

    if np.sign(db) == np.sign(old_db):
        learning_rate_b *= 1.1
    else:
        learning_rate_b *= 0.5

    max_lr = 0.1
    min_lr = 1e-6  # 0.000001

    learning_rate_w = np.clip(learning_rate_w, min_lr, max_lr)
    learning_rate_b = np.clip(learning_rate_b, min_lr, max_lr)
    return learning_rate_w, learning_rate_b

#endregion



W, b = train(final_train_set_x, final_train_set_y, num_iterations=4000, learning_rate=0.005, is_adaptive=True)
Y_prediction_test = predict(final_test_set_x, W, b)
Y_prediction_train = predict(final_train_set_x, W, b)
# Print train/test Errors
print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - final_train_set_y)) * 100))
print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - final_test_set_y)) * 100))
# final tests