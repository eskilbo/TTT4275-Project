from logging import logProcesses
from os import PRIO_PGRP
import numpy as np
from numpy.core.fromnumeric import argmax, reshape, transpose
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import matplotlib.pyplot as plt
from scipy.spatial import distance

from PIL import Image

N_TRAIN = 60000
N_TEST = 10000
# nClasses = 10
# nClusters = 64
# ourK = 7

with open('MNIST/Data/train_images.bin','rb') as binaryFile:
    train_img = binaryFile.read()
with open('MNIST/Data/train_labels.bin','rb') as binaryFile:
    train_labels = binaryFile.read()
with open('MNIST/Data/test_images.bin','rb') as binaryFile:
    test_img = binaryFile.read()
with open('MNIST/Data/test_labels.bin','rb') as binaryFile:
    test_labels = binaryFile.read()

train_img = np.reshape(np.frombuffer(train_img[16:16+784*N_TRAIN], dtype=np.uint8), (N_TRAIN,784))
train_labels = np.frombuffer(train_labels[8:N_TRAIN+8], dtype=np.uint8)
test_img= np.reshape(np.frombuffer(test_img[16:16+784*N_TEST], dtype=np.uint8), (N_TEST,784))
test_labels = np.frombuffer(test_labels[8:N_TEST+8], dtype=np.uint8)


def NN():
    n_chunk = 10
    template_size = N_TEST//n_chunk #6000
    val_size = N_TEST//n_chunk #1000
    predicted_mat = np.zeros((10,val_size))
    actual_mat = np.zeros((10,val_size))
    predicted = np.zeros((10,1))
    actual = np.zeros((10,1))
    for chunk in range(n_chunk):
        template_img = train_img[chunk*template_size:(chunk+1)*template_size]
        template_lab = train_labels[chunk*template_size:(chunk+1)*template_size]
        val_img = test_img[chunk*val_size:(chunk+1)*val_size]
        val_lab = test_labels[chunk*val_size:(chunk+1)*val_size]

        print(f"chunk {chunk}/9")
        for i in range(val_size): 
            max = 0
            actual_mat[chunk][i] = val_lab[i]
            actual[val_lab[i]] += 1

            prediction = 0
            min = 255**2*28*28
            for j in range(template_size):
                d = distance.euclidean(val_img[i],template_img[j])
                if d<min:
                    min = d
                    prediction = template_lab[j]
                    predicted_mat[chunk][i] = template_lab[j]
            if prediction == val_lab[i]:
                predicted[prediction] += 1
    print(f"actual: {actual}\n\n predicted: {predicted}")
    return

    





def main():
    # img = np.reshape(train_img[0],(28,28))
    # print(train_labels[0])
    # plt.imshow(img)
    # plt.show()

    NN()

    return

if __name__=='__main__':
    main()
