import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sn
from PIL import Image

"""
########################
    ML Functions 
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


def split_dataset(X, y, test_ratio=0.3, random_state=None):
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
########################
    SVM & SMO Functions 
########################

"""

"""
Description: standardize features

Parameters:
    x - dataframe of features
Returns:
    x - std of x
"""


def std_features(x):
    x = (x - x.mean()) / (x.std())
    return x


"""
Description: standardize labels

Parameters:
    y - array of labels
Returns:
    y - std of y
"""


def std_label(y):
    for i in range(y.shape[0]):
        if y[i] == 'M':
            y[i] = 1
        else:
            y[i] = -1
    return y


"""
Description: Select alpha

Parameters:
    i - alpha
    m - the number of alpha
Returns:
    j -
"""


def select_jrand(i, m):
    j = i
    while (j == i):
        j = int(np.random.uniform(0, m))
    return j


"""
Description: Clip alpha

Parameters:
    aj - alpha value
    H - alpha upper limit
    L - alpha lower limit
Returns:
    aj - alpah value
"""


def clip_alpha(aj, H, L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj


"""
Description: Simplified SMO algorithm

Parameters:
    dataMatIn - features matrix
    classLabels - labels matrix
    C - penalty
    toler - slack variable
    epoch - maximum number of iteration
Returns:
    b - threshold
    alphas -
"""


def simple_smo(dataMatIn, classLabels, C, toler, epoch=50):
    # Transform to numpy array
    dataMatrix = np.mat(dataMatIn);
    labelMat = np.mat(classLabels).transpose()
    # Initialize b, shape of matrix
    b = 0;
    m, n = np.shape(dataMatrix)
    # Initialize alphas
    alphas = np.mat(np.zeros((m, 1)))
    # Move to training process
    iter_num = 0
    while (iter_num < epoch):
        alphaPairsChanged = 0
        for i in range(m):
            # Step 1：Compute Error Ei
            fXi = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[i, :].T)) + b
            Ei = fXi - float(labelMat[i])
            # Optimize alpha with penalty。
            if ((labelMat[i] * Ei < -toler) and (alphas[i] < C)) or ((labelMat[i] * Ei > toler) and (alphas[i] > 0)):
                # Randomly generate alpha_j  compared with alpha_i
                j = select_jrand(i, m)
                # Compute Error Ei
                fXj = float(np.multiply(alphas, labelMat).T * (dataMatrix * dataMatrix[j, :].T)) + b
                Ej = fXj - float(labelMat[j])
                # Update alphas
                alphaIold = alphas[i].copy();
                alphaJold = alphas[j].copy();
                # Step 2: Compute upper and lower boundary
                if (labelMat[i] != labelMat[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L == H:
                    print("L==H")
                    continue
                # Step 3: compute eta
                eta = 2.0 * dataMatrix[i, :] * dataMatrix[j, :].T - dataMatrix[i, :] * dataMatrix[i, :].T - dataMatrix[
                                                                                                            j,
                                                                                                            :] * dataMatrix[
                                                                                                                 j, :].T
                if eta >= 0:
                    print("eta>=0")
                    continue
                # Step 4: Update alpha_j
                alphas[j] = np.subtract(alphas[j], labelMat[j] * (Ei - Ej) / eta)
                # Step 5: Clip alpha_j
                alphas[j] = clip_alpha(alphas[j], H, L)
                if (abs(alphas[j] - alphaJold) < 0.00001):
                    print("alpha_j do not change")
                    continue
                # Step 6: Update alpha_i
                alphas[i] += labelMat[j] * labelMat[i] * (alphaJold - alphas[j])
                # Step 7: Update b_1 and b_2
                b1 = b - Ei - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[i, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[i, :] * dataMatrix[j, :].T
                b2 = b - Ej - labelMat[i] * (alphas[i] - alphaIold) * dataMatrix[i, :] * dataMatrix[j, :].T - \
                     labelMat[j] * (alphas[j] - alphaJold) * dataMatrix[j, :] * dataMatrix[j, :].T
                # Step 8：Select b according to b_1 and b_2
                if (0 < alphas[i]) and (C > alphas[i]):
                    b = b1
                elif (0 < alphas[j]) and (C > alphas[j]):
                    b = b2
                else:
                    b = (b1 + b2) / 2.0
                # 统计优化次数
                alphaPairsChanged += 1
                # 打印统计信息
                print("Epoch:%d Sample:%d, num of alpha:%d" % (iter_num, i, alphaPairsChanged))
        # Update iter_num
        if (alphaPairsChanged == 0):
            iter_num += 1
        else:
            iter_num = 0
        print("Total Epoch: %d" % iter_num)
    return b, alphas


"""
Description: compute w

Parameters:
    dataMat - features matrix
    labelMat - labels matrix
    alphas - alphas
Returns:
    w - hyperplane param
"""


def get_w(dataMat, labelMat, alphas):
    alphas, dataMat, labelMat = np.array(alphas), np.array(dataMat), np.array(labelMat)
    m, n = dataMat.shape
    w = np.zeros((n, 1))
    for i in range(m):
        w += np.multiply(alphas[i] * labelMat[i], dataMat[i, :].T)
    return w


"""
########################
  Visualize Functions 
########################
"""


def show_table(data, col_label):
    plt.axis('tight')
    plt.axis('off')
    plt.table(cellText=data, colLabels=['value', 'value'], rowLabels=col_label, loc="center")

    plt.show()


def plot_cost(J_all, num_epochs):
    plt.xlabel('Epochs')
    plt.ylabel('Cost')
    plt.plot(range(num_epochs, 0), J_all, 'm', linewidth="5")
    plt.show()


def plot_data(x, y):
    plt.xlabel('sqft living')
    plt.ylabel('price')
    plt.plot(x[:, 1], y, 'bo')
    plt.show()


def plot_result(x, y, X_train, pred):
    plt.xlabel('sqft living')
    plt.ylabel('price')
    plt.scatter(x[:, 1], y, color='blue')
    plt.plot(X_train[:, 1], pred, color='red')
    plt.show()


def plot_swarm(X, y):
    sns.set(style="whitegrid", palette="muted")
    data = X
    data_n_2 = (data - data.mean()) / (data.std())  # standardization
    data = pd.concat([y, data_n_2.iloc[:, 0:10]], axis=1)
    data = pd.melt(data, id_vars="diagnosis",
                   var_name="features",
                   value_name='value')
    plt.figure(figsize=(10, 10))
    sns.swarmplot(x="features", y="value", hue="diagnosis", data=data)

    plt.xticks(rotation=90)
    plt.show()


def show_results(y_test, y_predict):
    # True positive, false positive, true negative, false negative
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    # Compute the results
    for i in range(len(y_test)):
        if y_test[i]==y_predict[i]==1:
            TP += 1
        if y_test[i]==-1 and y_predict[i]==1:
            FP += 1
        if y_test[i]==y_predict[i]==-1:
            TN += 1
        if y_test[i]==1 and y_predict[i]==-1:
            FN += 1
    return TP, FP, FN, TN


def confusion_martix(array, row, col):
    df_cm = pd.DataFrame(array, index=[i for i in row],
                         columns=[i for i in col])
    # sn.set(font_scale=1.4)
    sn.heatmap(df_cm, annot=True, annot_kws={"size": 8})
    plt.show()