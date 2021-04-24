import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

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
C = 3 # Number of classes
F = 4 # Number of features

# Load data
data = datasets.load_iris()['data']
class1 = data[0:50]
print(class1)
class2 = data[50:100]
class3 = data[100:150]
classes = [class1, class2, class3]

# Split data into training and test sets
train1_1, test1_1 = train_test_split(class1, test_size=20, train_size=30, random_state=0, shuffle=False)
train2_1, test2_1 = train_test_split(class2, test_size=20, train_size=30, random_state=0, shuffle=False)
train3_1, test3_1 = train_test_split(class3, test_size=20, train_size=30, random_state=0, shuffle=False)
print(train1_1)
# Parameters for linear classifier
alpha = 0.01
iterations = 2000

#def linear_classifier()
