from SVM import SVM
import reg_utils


if __name__ == '__main__':
    # Load dataset
    dataset = reg_utils.load_dataset('dataset/breast_cancer_dataset.csv')

    # Split into features and labels
    x = dataset.drop(['id','diagnosis','Unnamed: 32'], axis = 1)
    y = dataset['diagnosis']

    # Visualize data
    reg_utils.plot_swarm(x, y)

    # Standardize x, y
    x = reg_utils.std_features(x).values
    y = reg_utils.std_label(y.values)
    # Train the SVM based on SMO
    X_train, X_test, y_train, y_test = reg_utils.split_dataset(x, y, test_ratio=0.3, random_state=13)
    # Regularization parameter, Tolerance, limit on iterations
    C, toler, epoch = 1, 0.001, 100
    linear_svm = SVM(X_train, y_train, max_iter=epoch, C=C, tolerance=toler)
    linear_svm.fit()

    print(linear_svm.predict(X_test[0,:]))
    print(y_test[0])
    print(y_test)
    # Predict test set
    total = y_test.shape[0]
    y_predict = []
    for i in range(total):
        y_hat = linear_svm.predict(X_test[i,:])
        y_predict.append(y_hat)

    print(y_predict)
    # Visualize Results
    TP, FP, FN, TN = reg_utils.show_results(y_test, y_predict)
    confusion_array = [[TP, FP], [FN, TN]]
    print("Precision:{}".format(TP / (TP + FP)))
    print("Recall:{}".format(TP / (TP + FN)))
    row = ['Pos', 'Neg']
    col = ['Pos', 'Neg']
    reg_utils.confusion_martix(confusion_array, row, col)