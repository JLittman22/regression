import pandas as pd
import numpy as np

def run_algorithm(df, num_folds, learning_rate):
    fold_splits = k_fold_cross_validation_split(df, num_folds)

    training_error_list = []
    test_error_list = []

    for i in range(num_folds):
        fold_splits_copy = list(fold_splits)
        test_df = fold_splits_copy.pop(i)
        training_df = pd.concat(fold_splits_copy)

        training_df, test_df = center_data(training_df, test_df)

        weights = calculate_weights(training_df, learning_rate)
        bias_term = np.mean(training_df.iloc[:, -1])

        training_error = evaluate_performance(training_df, weights, bias_term)
        test_error = evaluate_performance(test_df, weights, bias_term)

        training_error_list.append(training_error)
        test_error_list.append(test_error)
        print (weights)

    average_training_error = np.mean(np.asarray(training_error_list))
    average_test_error = np.mean(np.asarray(test_error_list))
    print ("Average Training Accuracy is: " + str(average_training_error))
    print ("Average Test Accuracy is: " + str(average_test_error))


def k_fold_cross_validation_split(data, num_folds):
    data_copy = data.copy()
    data_copy = data_copy.sample(frac=1)
    fold_splits = []

    fold_size = data.shape[0] // num_folds

    for fold in range(num_folds):
        index_offset = fold * fold_size
        df = pd.DataFrame(data_copy.iloc[index_offset:index_offset+fold_size])
        fold_splits.append(df)

    return fold_splits

# Returns Updated Training and Test Data
def center_data(training_df, test_df):
    feature_averages = {}

    for column_idx in training_df.iloc[:, :training_df.shape[1] - 1]:
        column = training_df[column_idx]
        col_mean = np.mean(column)
        feature_averages[column_idx] = col_mean

        for i in range(len(training_df[column_idx])):
            column.values[i] = column.values[i] - col_mean
        training_df[column_idx] = column

    for column_idx in test_df.iloc[:, :test_df.shape[1] - 1]:
        column = test_df[column_idx]
        col_mean = feature_averages[column_idx]

        for i in range(len(test_df[column_idx])):
            column.values[i] = column.values[i] - col_mean

        test_df[column_idx] = column

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

        if predicted > 0.5:
            predicted = 1
        else:
            predicted = 0

        if predicted == actual:
            sum += 1

    acc = sum / float(total_rows)

    return acc


def predict(row, weights):
    value = 0
    for i in range(len(row)):
        value += row[i] * weights[i]

    return value


df = pd.read_csv('/Users/justinlittman/dev/courses/DS4420/HW2/spambase.csv', header=None)

learning_rate = 0.1
num_folds = 3

run_algorithm(df, num_folds, learning_rate)