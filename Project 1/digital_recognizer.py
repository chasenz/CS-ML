import knn_utils
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Load dataset
    train_dataset = knn_utils.load_dataset('dataset/digital/train.csv')
    # train_dataset = knn_utils.load_dataset('dataset/digital/sample_submission.csv')
    # test_dataset = knn_utils.load_dataset('dataset/digital/test.csv')
    test_dataset = knn_utils.load_dataset('dataset/digital/sample_submission.csv')

    # Split dataset
    X_train = train_dataset.drop('label', axis=1).values
    X_test = test_dataset.values
    y_train = train_dataset['label'].values
    # y_test = test_dataset['label'].T.values

    # print(X_train)
    # print(len(X_train[:,0]))
    # knn_utils.show_img(X_train[:,0].reshape(28, 28).astype("int"),y_train[0])
    # knn_utils.plt_digital(X_train, y_train, 2, 2)
    # print(y_train)

    # # Features scaling
    # X_train = knn_utils.features_scaling(X_train)
    # X_test = knn_utils.features_scaling(X_test)
    #
    # Tran dataset based on KNN
    y_predict = knn_utils.KNN_classification(X_test, X_train, y_train, 1)

    print(y_predict)
    # Evaluate
    # eva = np.sum(np.abs(y_predict - y_test))
    # print(eva / len(y_predict))
    image_id = np.arange(1,len(y_predict) + 1)
    array = np.row_stack((image_id, np.array(y_predict))).T
    print(array)
    predict_pd = pd.DataFrame(array, columns=["ImageId","Label"])
    predict_pd.to_csv('Result.csv', index=False)
    #
    # # Algorithm results
    # confusion_array = knn_utils.pima_show_results(y_test, y_predict)
    # row = ['Pos','Neg']
    # col = ['Pos','Neg']
    # knn_utils.confusion_martix(confusion_array, row, col)
    #
    #
    # # Visualize dataset
    # knn_utils.hist_dataset(dataset)
    # knn_utils.heatmap_dataset(dataset)
