import numpy as np
from numpy.core.fromnumeric import argmax, reshape
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import DataFrame

from confusion_matrix_pretty_print import _test_cm, pretty_plot_confusion_matrix

# Tasks:
# 1a. Choose the first 30 samples for training and the last 20 for testing.
# 1b. Train a linear classifier as described in 2.4 and 3.2. Tune alpha until the training coverge.
# 1c. Find the confusion matrix and error rate for both sets.
# 1d. Use the last 30 samples for training and first 20 samples for test. Repeat the training and test phases for this case.
# 1e. Compare the results for the two cases and comment.
# 2a. Use first 30 for training and last 20 for test. Produce histograms for each feature and class.
#     Take away the feature which shows most overlap between classes. Train and test a classifier with the remaining three features.
# 2b. Repeat with respectively two and one features.
# 2c. Compare confusion matrices and error rates for the four experiments.
#     Comment on the property of the features with respect to linear separability both as a whole and for the three separate classes.

# Constants
N_CLASS = 3  # Number of classes
N_FEAT = 4  # Number of features

# Parameters for linear classifier
alpha = 0.01
N_BATCH = 30    # Batch size
N_ITER = 2000   # Number of interations
train_size = 30  # Number of training samples for each class
test_size = 20      # Number of test samples for each class    

### Load data ###
irisdata = datasets.load_iris()['data']
class1 = irisdata[0:50]
class2 = irisdata[50:100]
class3 = irisdata[100:150]

# Splitting the three data sets into training and test sets
train1, test1 = train_test_split(class1, test_size=test_size, train_size=train_size, random_state=0, shuffle=False)
train2, test2 = train_test_split(class2, test_size=test_size, train_size=train_size, random_state=0, shuffle=False)
train3, test3 = train_test_split(class3, test_size=test_size, train_size=train_size, random_state=0, shuffle=False)
train = np.concatenate((train1,train2,train3),axis=None)
train = np.reshape(train,[N_BATCH*N_CLASS,N_FEAT])

# Creating targets for each class
target1 = np.tile([1,0,0],N_BATCH)
target2 = np.tile([0,1,0],N_BATCH)
target3 = np.tile([0,0,1],N_BATCH)
target = np.concatenate((target1,target2,target3),axis=None)
target = np.reshape(target,[N_BATCH*N_CLASS,N_CLASS])


# Sigmoid function
def sigmoid(x):
    return np.array(1/(1+np.exp(-x)))

# Get confidence vectors for each observation
def forward_propagate(x_vec, W):
    g_vec = np.zeros([N_BATCH*N_CLASS,N_CLASS])
    for i,x in enumerate(x_vec):
        x = np.append([x],[1])
        z = W @ x
        g_vec[i] = sigmoid(z)
    return g_vec

# get update matrix
def get_mse_derivative(train,g_vec,target):
    dmse = np.zeros([N_CLASS,N_FEAT+1])
    for xk,gk,tk in zip(train,g_vec,target):    
        xk = np.append([xk],[1])
        xk = xk.reshape(N_FEAT+1,1)

        # temp = np.multiply(gk-tk,gk).reshape(N_CLASS,1)
        # temp = np.multiply(temp,np.ones((N_CLASS,1))-gk.reshape(N_CLASS,1))
        # dmse += temp @ xk.reshape(1,N_FEAT+1)

        dmse += (((gk-tk)*gk).reshape(N_CLASS,1) * (np.ones((N_CLASS,1))-gk.reshape(N_CLASS,1))) @ xk.reshape(1,N_FEAT+1)

    return dmse


def train_linear_classifier(W,iterations,step_size):
    loss = np.zeros([iterations,1],dtype=float)
    for i in range(iterations):
        g_vec = forward_propagate(train, W)
        dmse = get_mse_derivative(train,g_vec,target)
        loss[i] = mean_squared_error(g_vec,target)
        W = W - step_size*dmse
        print(f"Training iteration: {i}")
    return W, loss

def get_confusion_matrix(W,data_set,N_CLASS,N_BATCH):
    confusion_mat = np.zeros([N_CLASS,N_CLASS])

    predicted = forward_propagate(data_set, W)
    row = -1
    for i,gk in enumerate(predicted):
        if i%N_BATCH == 0:
            row += 1
        col = np.argmax(gk)
        confusion_mat[row][col] += 1
    return confusion_mat

def main():
    #Generate empty weights
    W = np.zeros([N_CLASS, N_FEAT+1],dtype=float)
    # Train the weights
    W, loss = train_linear_classifier(W,N_ITER,alpha)
    print(W)
    # Compute confusion matrix
    confusion_train = get_confusion_matrix(W,train,N_CLASS,N_BATCH)
    print(confusion_train)

    #pretty print confusion mat
    df_cm = DataFrame(confusion_train, index=["setosa","versicolor","virginica"], columns=["setosa","versicolor","virginica"])

    pretty_plot_confusion_matrix(df_cm,title='Training set',cmap="Oranges")
    return


if __name__ == '__main__':
    main()