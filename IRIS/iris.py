from logging import logProcesses
import numpy as np
from numpy.core.fromnumeric import argmax, reshape, transpose
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import matplotlib.pyplot as plt

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
N_ITER = 10000   # Number of interations
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
train = np.reshape(train,[train_size*N_CLASS,N_FEAT])
test = np.concatenate((test1,test2,test3),axis=None)
test = np.reshape(test,[test_size*N_CLASS,N_FEAT])


# Creating targets for each class
target1 = np.tile([1,0,0],train_size)
target2 = np.tile([0,1,0],train_size)
target3 = np.tile([0,0,1],train_size)
target = np.concatenate((target1,target2,target3),axis=None)
target = np.reshape(target,[train_size*N_CLASS,N_CLASS])


# Sigmoid function
def sigmoid(x):
    return np.array(1/(1+np.exp(-x)))

# Get confidence vectors for each observation
def forward_propagate(x_vec, W,n_batch):
    g_vec = np.zeros([n_batch*N_CLASS,N_CLASS])
    for i,x in enumerate(x_vec):
        x = np.append([x],[1])
        z = W @ x
        g_vec[i] = sigmoid(z)
    return g_vec

# get update matrix
def get_mse_derivative(train,n_feat,g_vec,target):
    dmse = np.zeros([N_CLASS,n_feat+1])
    for xk,gk,tk in zip(train,g_vec,target):    
        xk = np.append([xk],[1])
        xk = xk.reshape(n_feat+1,1)

        dmse += (((gk-tk)*gk).reshape(N_CLASS,1) * (np.ones((N_CLASS,1))-gk.reshape(N_CLASS,1))) @ xk.reshape(1,n_feat+1)
    return dmse


def train_linear_classifier(data,W,iterations,step_size,n_batch):
    loss = np.zeros([iterations,1],dtype=float)
    for i in range(iterations):
        g_vec = forward_propagate(data, W,n_batch)
        dmse = get_mse_derivative(data,len(data[0]),g_vec,target)
        loss[i] = mean_squared_error(g_vec,target)
        W = W - step_size*dmse
        print(f"Training iteration: {i}")
    return W, loss

def get_confusion_matrix(W,data_set,N_CLASS,N_BATCH):
    confusion_mat = np.zeros([N_CLASS,N_CLASS])

    predicted = forward_propagate(data_set, W,N_BATCH)
    row = -1
    for i,gk in enumerate(predicted):
        if i%N_BATCH == 0:
            row += 1
        col = np.argmax(gk)
        print(row)
        confusion_mat[row][col] += 1
    return confusion_mat

def plot_alphas():
    alphas = [0.05,0.01,0.005]
    for a in alphas:
        W = np.zeros([N_CLASS, N_FEAT+1],dtype=float)
        W,loss = train_linear_classifier(W,N_ITER,a,N_BATCH)
        #plot loss
        plt.plot(np.arange(N_ITER),loss,label="alpha= "+str(a))
        plt.legend()
        plt.xlabel("iterations")
        plt.ylabel("MSE")
    plt.show()


def task2_remove_feature(train,test,N_CLASS,N_FEAT,alpha,N_ITER,train_size,test_size):

    # Remove feature "sepal width"
    W = np.zeros([N_CLASS, N_FEAT],dtype=float)
    train = np.delete(train,1,1)
    test = np.delete(test,1,1)
    W, _ = train_linear_classifier(train,W,N_ITER,alpha,train_size)
    conf1_train = get_confusion_matrix(W,train,N_CLASS,train_size)
    conf1_test = get_confusion_matrix(W,test,N_CLASS,test_size)

    # Remove feature "sepal length"
    W = np.zeros([N_CLASS, N_FEAT-1],dtype=float)
    train = np.delete(train,0,1)
    test = np.delete(test,0,1)
    W, _ = train_linear_classifier(train,W,N_ITER,alpha,train_size)
    conf2_train = get_confusion_matrix(W,train,N_CLASS,train_size)
    conf2_test = get_confusion_matrix(W,test,N_CLASS,test_size)

    # Remove feature "petal length"
    W = np.zeros([N_CLASS, N_FEAT-2],dtype=float)
    train = np.delete(train,0,1)
    test = np.delete(test,0,1)
    W, _ = train_linear_classifier(train,W,N_ITER,alpha,train_size)
    conf3_train = get_confusion_matrix(W,train,N_CLASS,train_size)
    conf3_test = get_confusion_matrix(W,test,N_CLASS,test_size)

    # Plotting confusion matrices
    conf_arr = [[conf1_train,conf1_test],[conf2_train,conf2_test],[conf3_train,conf3_test]]
    labels = ["sepal width","sepal width, sepal length","sepal width, sepal length, petal length"]
    for i, (train, test) in enumerate(conf_arr):

        df_cm = DataFrame(train, index=["setosa","versicolor","virginica"], columns=["setosa","versicolor","virginica"])
        pretty_plot_confusion_matrix(df_cm,title='Training set - Removed '+str(labels[i]),cmap="Oranges",pred_val_axis='x')

        df_cm = DataFrame(test, index=["setosa","versicolor","virginica"], columns=["setosa","versicolor","virginica"])
        pretty_plot_confusion_matrix(df_cm,title='Test set - Removed '+str(labels[i]),cmap="Oranges",pred_val_axis='x')
    return 


def main():
    ### Task 1 ###

    #Generate empty weights
    W = np.zeros([N_CLASS, N_FEAT+1],dtype=float)
    
    # Train linear classifier
    W, loss = train_linear_classifier(train,W,N_ITER,alpha,N_ITER,train_size,test_size)
    
    # Compute confusion matrices
    confusion_train = get_confusion_matrix(W,train,N_CLASS,train_size)
    confusion_test = get_confusion_matrix(W,test,N_CLASS,test_size)

    # Plot confusion matrix for training set
    df_cm = DataFrame(confusion_train, index=["setosa","versicolor","virginica"], columns=["setosa","versicolor","virginica"])
    pretty_plot_confusion_matrix(df_cm,title='Training set',cmap="Oranges",pred_val_axis='x')
    # Plot confusion matrix for test set
    df_cm = DataFrame(confusion_test, index=["setosa","versicolor","virginica"], columns=["setosa","versicolor","virginica"])
    pretty_plot_confusion_matrix(df_cm,title='Test set',cmap="Oranges",pred_val_axis='x')


    ### Task 2 ###

    task2_remove_feature(train,test,N_CLASS,N_FEAT,alpha,N_ITER,train_size,test_size)
     
    return


if __name__ == '__main__':
    main()