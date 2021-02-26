import knn_utils
import numpy as np
import pandas as pd

if __name__ == '__main__':
    # Load dataset
    train_dataset = knn_utils.load_dataset('dataset/digital/train.csv')
    test_dataset = knn_utils.load_dataset('dataset/digital/test.csv')

    # Split into (X,y)
    X_train = train_dataset.drop('label', axis=1).values
    X_test = test_dataset.values
    y_train = train_dataset['label'].values

    # Train dataset based on KNN
    y_predict = knn_utils.KNN_classification(X_test, X_train, y_train, 1)

    # Save results
    image_id = np.arange(1,len(y_predict) + 1)
    array = np.row_stack((image_id, np.array(y_predict))).T
    predict_pd = pd.DataFrame(array, columns=["ImageId","Label"])
    predict_pd.to_csv('Result.csv', index=False)


    # Visualize algorithm results using confusion matrix
    # My training result
    test_result = knn_utils.load_dataset('dataset/digital/test_result_k_3.csv')
    # Answer with 100% accuracy
    test_answer = knn_utils.load_dataset('dataset/digital/test_answer.csv')
    y_test = test_answer['Label'].values
    y_predict = test_result['Label'].values
    confusion_array = knn_utils.digital_show_results(y_test, y_predict)
    print(confusion_array[8][8])
    row = '0123456789'
    col = '0123456789'
    knn_utils.confusion_martix(confusion_array, row, col)

