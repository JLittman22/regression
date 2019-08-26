import pandas as pd
import numpy as np

def run_algorithm(training_df, test_df, epochs, learning_rate):
    training_df = normalize_data(training_df)
    test_df = normalize_data(test_df)

    training_df_copy = training_df.copy()

    training_df_copy.insert(0, 'intercept', 1)

    weights = calculate_weights(training_df_copy, epochs, learning_rate)

    training_error = evaluate_performance(training_df, weights)
    test_error = evaluate_performance(test_df, weights)
    print (weights)
    print ("Training MSE is: " + str(training_error))
    print ("Test MSE is: " + str(test_error))


def normalize_data(data):
    for column in data.iloc[:, :data.shape[1] - 1].columns:
        min = np.amin(data[column])
        max = np.amax(data[column])
        for i in range(len(data[column])):
            value = (data[column][i] - min) / float(max)
            data.at[i, column] = value

    return data


def calculate_weights(df, epochs, learning_rate):
    X = df.iloc[:, :df.shape[1] - 1]
    Y = df.iloc[:, -1]

    weights = np.zeros(X.shape[1])

    for epoch in range(epochs):
        predictions = X.dot(weights.T)
        errors = predictions - Y

        if epoch % 10 == 0:
            print(np.mean(np.square(errors)))

        gradients = []
        for column in X.columns:
            x_values = X[column]
            gradient = errors.dot(x_values.values)
            gradients.append(gradient)

        weights = weights - learning_rate * np.array(gradients)

    return weights


def evaluate_performance(df, weights):
    total_rows = df.shape[0]
    sum = 0
    for idx, row in df.iterrows():
        predicted = predict(row[:df.shape[1]-1], weights)
        actual = row.iloc[-1]

        sum += (predicted - actual) ** 2

    mse = sum / float(total_rows)

    return mse


def predict(row, weights):
    value = row.dot(weights[1:]) + weights[0]
    return value


training_df = pd.read_csv('/Users/justinlittman/dev/courses/DS4420/HW2/housing_train.txt', header=None, sep=' +', engine='python')
test_df = pd.read_csv('/Users/justinlittman/dev/courses/DS4420/HW2/housing_test.txt', header=None, sep=' +', engine='python')

learning_rate = 0.0001
epochs = 4000

run_algorithm(training_df, test_df, epochs, learning_rate)