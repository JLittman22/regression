import pandas as pd
import numpy as np

def run_algorithm(training_df, test_df, learning_rate):
    training_df, test_df = center_data(training_df, test_df)

    weights = calculate_weights(training_df, learning_rate)
    bias_term = np.mean(training_df.iloc[:, -1])

    training_error = evaluate_performance(training_df, weights, bias_term)
    test_error = evaluate_performance(test_df, weights, bias_term)
    print (weights)
    print ("Training MSE is: " + str(training_error))
    print ("Test MSE is: " + str(test_error))


# Returns Updated Training and Test Data
def center_data(training_df, test_df):
    feature_averages = {}

    for column_idx in training_df.iloc[:, :training_df.shape[1] - 1]:
        column = training_df[column_idx]
        col_mean = np.mean(column)
        feature_averages[column_idx] = col_mean

        for i in range(len(training_df[column_idx])):
            value = training_df[column_idx][i] - col_mean
            training_df.at[i, column_idx] = value

    for column_idx in test_df.iloc[:, :test_df.shape[1] - 1]:
        col_mean = feature_averages[column_idx]

        for i in range(len(test_df[column_idx])):
            value = test_df[column_idx][i] - col_mean
            test_df.at[i, column_idx] = value

    return training_df, test_df


def calculate_weights(df, learning_rate):
    lr = learning_rate
    X = df.iloc[:, :df.shape[1] - 1]
    Y = df.iloc[:, -1]

    X_T_mult_X_df = X.T.dot(X)
    df_inverse = pd.DataFrame(np.linalg.pinv(X_T_mult_X_df.values + lr * np.eye(X_T_mult_X_df.shape[0], X_T_mult_X_df.shape[1])))
    weights = df_inverse.dot(X.T.values).dot(Y.values)

    return weights


def evaluate_performance(df, weights, bias):
    total_rows = df.shape[0]
    sum = 0
    for idx, row in df.iterrows():
        predicted = predict(row[:df.shape[1]-1], weights) + bias
        actual = row.iloc[-1]

        sum += (predicted - actual) ** 2

    mse = sum / float(total_rows)

    return mse


def predict(row, weights):
    value = 0
    for i in range(len(row)):
        value += row[i] * weights[i]

    return value


training_df = pd.read_csv('/Users/justinlittman/dev/courses/DS4420/HW2/housing_train.txt', header=None, sep=' +', engine='python')
test_df = pd.read_csv('/Users/justinlittman/dev/courses/DS4420/HW2/housing_test.txt', header=None, sep=' +', engine='python')

learning_rate = 0

run_algorithm(training_df, test_df, learning_rate)