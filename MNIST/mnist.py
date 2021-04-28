from logging import logProcesses
from os import PRIO_PGRP
import numpy as np
from numpy.core.fromnumeric import argmax, reshape, transpose
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from pandas import DataFrame
import matplotlib.pyplot as plt
from PIL import Image
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
import time

def load_data(N_TRAIN, N_TEST):
    with open('Data/train_images.bin','rb') as binaryFile:
        train_img = binaryFile.read()
    with open('Data/train_labels.bin','rb') as binaryFile:
        train_labels = binaryFile.read()
    with open('Data/test_images.bin','rb') as binaryFile:
        test_img = binaryFile.read()
    with open('Data/test_labels.bin','rb') as binaryFile:
        test_labels = binaryFile.read()
    train_img = np.reshape(np.frombuffer(train_img[16:16+784*N_TRAIN], dtype=np.uint8), (N_TRAIN,784))
    train_labels = np.frombuffer(train_labels[8:N_TRAIN+8], dtype=np.uint8)
    test_img= np.reshape(np.frombuffer(test_img[16:16+784*N_TEST], dtype=np.uint8), (N_TEST,784))
    test_labels = np.frombuffer(test_labels[8:N_TEST+8], dtype=np.uint8)
    return train_img, train_labels, test_img, test_labels

def diff_image(img1, img2):
    a = img1-img2
    b = np.uint8(img1<img2) * 254 + 1
    return a * b

def euclid_distance(img1, img2):
    return np.sum(diff_image(img1, img2))

def nearest_neighbor_classifier(train_img, train_labels, test_img, test_labels, N_TRAIN, N_TEST):
    chunks = 1
    train_size = N_TRAIN//chunks
    test_size = N_TEST//chunks
    correct_pred = []
    failed_pred = []
    predicted = np.zeros((10,1))
    actual = np.zeros((10,1))
    confusion_matrix = np.zeros((10,10),dtype=int)
    start = time.time()
    for chunk in range(chunks):
        training_img = train_img[chunk*train_size:(chunk+1)*train_size]
        training_lab = train_labels[chunk*train_size:(chunk+1)*train_size]
        testing_img = test_img[chunk*test_size:(chunk+1)*test_size]
        testing_lab = test_labels[chunk*test_size:(chunk+1)*test_size]
        #print(f"CHUNK {chunk+1}/{chunks}")
        for i in range(test_size):
            print(f"Test image {i+1}/10000")
            actual[testing_lab[i]] += 1
            prediction = 0
            min = 65025*28*28*2
            pred_img_index = 0
            for j in range(train_size):
                d = euclid_distance(testing_img[i], training_img[j])
                if d < min:
                    min = d
                    pred_img_index = j
                    prediction = training_lab[j]
            if prediction == testing_lab[i]:
                predicted[prediction] += 1
                correct_pred.append([i, pred_img_index])
                confusion_matrix[testing_lab[i]][testing_lab[i]] += 1
            else:
                confusion_matrix[testing_lab[i]][prediction] += 1
                failed_pred.append([i, pred_img_index])
            
    end = time.time()
    print(f"Runtime of program is {(end-start)/60} minutes.")
    print(f"actual: {actual}\n\n predicted correctly: {predicted}")
    print(confusion_matrix)
    np.save("pred_correct.npy", np.array(correct_pred))
    np.save("pred_failed.npy", np.array(failed_pred))
    np.save("confusionmat.npy", confusion_matrix)
    df_cm = DataFrame(confusion_matrix, index=["0","1","2","3","4","5","6","7","8","9"], columns=["0","1","2","3","4","5","6","7","8","9"])
    pretty_plot_confusion_matrix(df_cm,title='Confusion matrix - 1NN Classifier without clustering - 10000 test 60000 train',cmap="Oranges",pred_val_axis='x')
    return 


def main():
    N_TRAIN = 60000
    N_TEST = 10000
    train_img, train_labels, test_img, test_labels = load_data(N_TRAIN, N_TEST)
    nearest_neighbor_classifier(train_img, train_labels, test_img, test_labels, N_TRAIN, N_TEST)
    return

if __name__=='__main__':
    main()