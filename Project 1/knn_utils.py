import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image
"""
########################
    KNN Functions 
########################
"""

"""
Description: Load dataset from the file

Parameters:
    filepath - the local path of the dataset file
Returns:
    dataset - the type of pandas dataset
"""
def load_dataset(filepath):
    # Load dateset from the file
    dataset = pd.read_csv(filepath)
    return dataset

"""
Description: Split dataset

Parameters:
    X - features array
    y - labels array
    test_ratio - the ratio of test set between 0 and 1
    random_state - the seed of random function
Returns:
    X_train - features of traning set
    X_test - features of test set
    y_train - labels of traning set
    y_test - labels of test set
"""
def split_dataset(X, y, test_ratio = 0.3, random_state = None):
    # Set random seed
    if random_state:
        np.random.seed(random_state)
    # Create random index to shuffle array
    shuffle_index = np.random.permutation(len(X))
    # Calculate the size of test set
    test_size = int(len(X) * test_ratio)
    # Split shuffle index
    shuffle_test = shuffle_index[:test_size]
    shuffle_train = shuffle_index[test_size:]
    # Split input arrays
    X_train = X[shuffle_train]
    X_test = X[shuffle_test]
    y_train = y[shuffle_train]
    y_test = y[shuffle_test]
    return X_train, X_test, y_train, y_test

"""
Description: Features scaling based on standardization

Parameters:
    X - the inital features value
Returns:
    std_scale - value after features scaling
"""
def features_scaling(X):
    # compute the mean
    mean = np.mean(X, axis=0)
    # compute standard deviation
    std = (np.sum((X - mean) ** 2, axis= 0) / X.shape[0]) ** 0.5
    # compute standardization
    return (X - mean) / std

"""
Description: Implement KNN classification

Parameters:
    X_test - features of test set
    X_train - features of traning set
    y_train - labels of traning set
    k - the value of k-nearest neighbors
Returns:
    y_predict - predicted labels of test set
"""
def KNN_classification(X_test, X_train, y_train, k):
    y_predict = []
    count = 0
    for test in X_test:
        # Calculate Euclid Distance
        # euclid_distance = np.sum((X_train - test) ** 2, axis= 1) ** 0.5
        distance = distance_helper(X_train, test, 3)
        # Sort distance
        distance_index = distance.argsort()
        # Count the num of labels
        count_labels = {}
        for i in range(k):
            label = y_train[distance_index[i]]
            count_labels[label] = count_labels.get(label,0) + 1
        # Find the maximum labels
        items = list(count_labels.items())
        items.sort(key=lambda x: x[1], reverse=True)
        # Add results to the prediction array
        y_predict.append(items[0][0])
        count = count + 1
        print(count)
    return y_predict

"""
Description: Compute distance using different methods

Parameters:
    X_train - features of traning set
    test - one test input
    method - Euclidean, Manhattan and Minkowski distance which represents in 1,2,3
Returns:
    distance - array contains the distance from all training points to the input point
"""
def distance_helper(X_train, test, method=1, p = 5):
    if method == 2:
        return np.sum(np.abs(X_train - test), axis= 1)
    if method == 3:
        return np.sum(np.abs(X_train - test) ** p, axis= 1) ** (1 / p)
    else:
        return np.sum((X_train - test) ** 2, axis= 1) ** 0.5


"""
########################
  Visualize Functions 
########################
"""
def hist_dataset(dataset):
    col_plot = dataset.hist(figsize=(20, 10))
    plt.show()

def heatmap_dataset(dataset):
    corr = dataset.corr()
    fig, ax = plt.subplots(figsize=(12, 9))
    sns.heatmap(corr, vmax=1, annot=True, square=True);
    plt.show()

def pima_show_results(y_test, y_predict):
    # True positive, false positive, true negative, false negative
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # Compute the results
    for i in range(len(y_test)):
        if y_test[i]==y_predict[i]==1:
            TP += 1
        if y_test[i]==0 and y_predict[i]==1:
            FP += 1
        if y_test[i]==y_predict[i]==0:
            TN += 1
        if y_test[i]==1 and y_predict[i]==0:
            FN += 1
    return [[TP, FP],
            [FN, TN]]

def digital_show_results(y_test, y_predict):
    # Initialize confusion matrix
    matrix = np.zeros((10,10), dtype=np.int32)
    for i in range(len(y_test)):
        if y_test[i] == y_predict[i]:
            matrix[y_test[i]][y_test[i]] = matrix[y_test[i]][y_test[i]] + 1
        else:
            matrix[y_test[i]][y_predict[i]] = matrix[y_test[i]][y_predict[i]] + 1
    return matrix


def confusion_martix(array, row, col):
    df_cm = pd.DataFrame(array, index=[i for i in row],
                         columns=[i for i in col])
    # sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
    plt.show()

def show_img(img_array, label, ax):
    img = Image.fromarray(img_array)
    # img_gray = img.convert("L")
    ax.imshow(img)
    ax.axis("on")
    ax.set_title("MNIST Image:{0}".format(label), fontsize=28)


def plt_digital(X_train, y_train, row, col):
    n = row * col
    fig, axes = plt.subplots(row, col, figsize=(20, 20))
    for i in range(n):
        r = i // col
        c = i % col
        ax = axes[r][c]
        show_img(X_train[:,i].reshape(28, 28).astype("int"), y_train[i], ax)
        ax.axis('off')
    plt.show()