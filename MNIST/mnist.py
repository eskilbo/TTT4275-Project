from logging import logProcesses
from os import PRIO_PGRP
import numpy as np
from numpy.core.fromnumeric import argmax, reshape, transpose
from sklearn import cluster
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from pandas import DataFrame
import matplotlib.pyplot as plt
from PIL import Image
from confusion_matrix_pretty_print import pretty_plot_confusion_matrix
from scipy.spatial import distance
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

def euclid_distance(img1, img2):
    a = img1-img2
    b = np.uint8(img1<img2) * 254 + 1
    return np.sum(a * b)

def nearest_neighbor_classifier(train_img, train_labels, test_img, test_labels, N_TRAIN, N_TEST):
    correct_pred = [] # Array of set of indeces of correct predictions
    failed_pred = [] # Array of set of indeces of incorrect predictions
    confusion_matrix = np.zeros((10,10),dtype=int)
    start = time.time()
    # Iterating through every test image
    for i in range(N_TEST):
        prediction = 0
        pred_img_index = 0
        min = float('inf')
        # Comparing distance of test image to every training image
        for j in range(N_TRAIN):
            #d = euclid_distance(test_img[i], train_img[j])
            d = distance.euclidean(test_img[i], train_img[j])
            if d < min:
                min = d
                pred_img_index = j
                prediction = train_labels[j]
        if prediction == test_labels[i]:
            correct_pred.append([i, pred_img_index])
            confusion_matrix[test_labels[i]][test_labels[i]] += 1
        else:
            confusion_matrix[test_labels[i]][prediction] += 1
            failed_pred.append([i, pred_img_index])
    end = time.time()
    print(f"Runtime of program is {(end-start)/60} minutes.")
    df_cm = DataFrame(confusion_matrix, index=["0","1","2","3","4","5","6","7","8","9"], columns=["0","1","2","3","4","5","6","7","8","9"])
    pretty_plot_confusion_matrix(df_cm,title='Confusion matrix - 1NN Classifier with clustering - 10000 test 640 clusters',cmap="Oranges",pred_val_axis='x')
    return 

def clustering(data,data_lab, n_clusters):
    N = 10
    # Sorting out classes from dataset
    templates_by_class = []
    for n in range(N):
        t = []
        for i,label in enumerate(data_lab):
            if n == label:
                t.append(data[i])
        templates_by_class.append(t)

    clusters = []
    labels = []
    for i in range(N):
        part = templates_by_class[i]
        kmeans = KMeans(n_clusters=n_clusters).fit(part)
        clusters.append(kmeans.cluster_centers_)
        labels.append([i]*n_clusters)

    cluster_img = np.array(clusters).flatten().reshape((n_clusters*N,data.shape[1]))
    cluster_lab = np.array(labels,dtype=int).flatten().reshape(n_clusters*N,1)
    np.save('cluster_img.npy',cluster_img)
    np.save('cluster_lab.npy',cluster_lab)
    return cluster_img, cluster_lab


def main():
    N_TRAIN = 60000
    N_TEST = 10000

    # Load data 
    template_img, template_labels, test_img, test_labels = load_data(N_TRAIN, N_TEST)
    
    # Run the clustering once, saves data into files
    #template_img, template_labels = clustering(train_img,train_labels,64)
    
    # If already clustered, load the files instead of running the clustering
    #template_img = np.load('cluster_img.npy')
    #template_labels = np.load('cluster_lab.npy')

    # Run nearest neighbor classifier
    nearest_neighbor_classifier(template_img, template_labels, test_img, test_labels, template_img.shape[0], N_TEST)
    return

if __name__=='__main__':
    main()