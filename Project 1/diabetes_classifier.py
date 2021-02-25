import knn_utils
import numpy as np

if __name__ == '__main__':
    # Load dataset
    dataset = knn_utils.load_dataset('dataset/diabetes.csv')

    # Split dataset
    X = dataset.drop('Outcome', axis=1).values
    y = dataset['Outcome'].values
    X_train, X_test, y_train, y_test = knn_utils.split_dataset(X,y,random_state=8)

    # Features scaling
    X_train = knn_utils.features_scaling(X_train)
    X_test = knn_utils.features_scaling(X_test)

    # Tran dataset based on KNN
    y_predict = knn_utils.KNN_classification(X_test, X_train, y_train, 1)

    print(y_predict)
    # Evaluate
    eva = np.sum(np.abs(y_predict - y_test))
    print(eva / len(y_predict))

    # Algorithm results
    confusion_array = knn_utils.pima_show_results(y_test, y_predict)
    row = ['Pos','Neg']
    col = ['Pos','Neg']
    knn_utils.confusion_martix(confusion_array, row, col)


    # Visualize dataset
    knn_utils.hist_dataset(dataset)
    knn_utils.heatmap_dataset(dataset)
