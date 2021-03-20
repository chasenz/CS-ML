import numpy as np
import reg_utils

"""
Description: Feature selection based on Pearson's Correlation

Parameters:
    X - features array
    y - labels array
Returns:
    r - correlation coefficient matrix of the features
"""


def feature_selection(X, y):
    cor_list = []
    col_size = X.shape[1]
    for i in range(col_size):
        cor = np.corrcoef(X[:, i], y)[0, 1]
        cor_list.append(cor)
    return cor_list


"""
Description: Model

Parameters:
    x - features array
    theta - parameters of the model
Returns:
    y - the result of theta * x
"""


def h(x, theta):
    return np.matmul(x, theta)


"""
Description: Cost function

Parameters:
    x - features array
    y - the result of theta * x
    theta - parameters of the model
Returns:
    the result of cost function
"""


def cost_function(x, y, theta):
    return ((h(x, theta) - y).T @ (h(x, theta) - y)) / (2 * y.shape[0])


"""
Description: Gradient descent 
"""


def gradient_descent(x, y, theta, learning_rate, num_epochs=10):
    m = x.shape[0]
    J_all = []

    for i in range(num_epochs):
        h_x = h(x, theta)
        dh_x = (1 / m) * (x.T @ (h_x - y))
        theta = theta - (learning_rate) * dh_x
        cost = cost_function(x, y, theta)[0, 0]
        J_all.append(cost)
        print("num epochs:{0}=====>cost function:{1}".format(i, cost))
    return theta, J_all


if __name__ == '__main__':
    # Load dataset
    dataset = reg_utils.load_dataset('dataset/kc_house_data.csv')

    # Split dataset
    X = dataset.drop(['price', 'id', 'date'], axis=1).values
    X_name = dataset.drop(['price', 'id', 'date'], axis=1).columns
    y = dataset['price'].values

    # Feature selection
    cor_list = feature_selection(X, y)
    max_index = cor_list.index(max(cor_list))
    max_feature = X_name[max_index]
    print(max_feature)
    # Draw the table
    cor_list_str = np.reshape(cor_list, (18, 1))
    reg_utils.show_table(cor_list_str, X_name.values.tolist())

    # Train the model
    # Split the dataset
    x = dataset['sqft_living'].values
    x = np.reshape(x, (x.shape[0], 1))
    y = np.reshape(y, (y.shape[0], 1))
    x = np.hstack((np.ones((x.shape[0], 1)), x))
    X_train, X_test, y_train, y_test = reg_utils.split_dataset(x, y, test_ratio=0.2)

    # Generate the linear regression model
    theta = np.zeros((X_train.shape[1], 1))
    learning_rate = 0.1
    num_epoachs = 50
    theta, J_all = gradient_descent(X_train, y_train, theta, learning_rate, num_epoachs)

    # Plot the result
    reg_utils.plot_result(X_train, y_train, X_train, theta[0][0] + theta[1][0] * X_train)
    reg_utils.plot_result(X_test, y_test, X_train, theta[0][0] + theta[1][0] * X_train)
