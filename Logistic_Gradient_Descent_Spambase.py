import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def run_algorithm(df, num_folds, epochs, learning_rate):
    df = normalize_data(df)

    fold_splits = k_fold_cross_validation_split(df, num_folds)

    training_set_accuracies = []
    test_set_accuracies = []

    for i in range(num_folds):
        fold_splits_copy = list(fold_splits)
        test_df = fold_splits_copy.pop(i)
        training_df = pd.concat(fold_splits_copy)

        training_df_copy = training_df.copy()

        training_df_copy.insert(0, 'intercept', 1)

        weights = calculate_weights(training_df_copy, epochs, learning_rate)

        training_error = evaluate_performance(training_df, weights)
        test_error = evaluate_performance(test_df, weights)

        training_set_accuracies.append(training_error)
        test_set_accuracies.append(test_error)
        print (weights)

        # confusion_matrix = get_confusion_matrix_values(training_df, weights)
        # print(confusion_matrix)

        auc = plot_roc_curve(test_df, weights)
        print("AUC is: " + str(auc))

    average_training_error = np.mean(np.asarray(training_set_accuracies))
    average_test_error = np.mean(np.asarray(test_set_accuracies))
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


def sigmoid(x):
    value = 1/(1+np.exp(-x))
    return value


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
        predictions = sigmoid(X.dot(weights.T))
        errors = Y - predictions

        if epoch % 10 == 0:
            print(np.mean(np.square(errors)))

        gradients = []
        for column in X.columns:
            x_values = X[column]
            gradient = errors.dot(x_values.values)
            gradients.append(gradient)

        weights = weights + learning_rate * np.array(gradients)

    return weights


def evaluate_performance(df, weights):
    total_rows = df.shape[0]
    sum = 0
    for idx, row in df.iterrows():
        predicted = sigmoid(predict(row[:df.shape[1]-1], weights))
        actual = row.iloc[-1]

        if predicted > 0.5:
            predicted = 1
        else:
            predicted = 0

        if predicted == actual:
            sum += 1

    acc = sum / float(total_rows)

    return acc

def plot_roc_curve(df, weights):
    true_positive_rates = [0, 1]
    false_positive_rates = [0, 1]

    for threshold in range(0, 101, 5):
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for idx, row in df.iterrows():
            predicted = sigmoid(predict(row[:df.shape[1]-1], weights))
            actual = row.iloc[-1]

            if predicted > threshold / float(100):
                predicted = 1
            else:
                predicted = 0

            if predicted == 1 and actual == 1:
                TP += 1
            elif predicted == 1 and actual == 0:
                FP += 1
            elif predicted == 0 and actual == 1:
                FN += 1
            elif predicted == 0 and actual == 0:
                TN += 1

            if TP == 0:
                TPR = 0
            else:
                TPR = TP / float(TP + FN)
            if FP == 0:
                FPR = 0
            else:
                FPR = FP / float(TN + FP)

        true_positive_rates.append(TPR)
        false_positive_rates.append(FPR)

    plt.plot(false_positive_rates, true_positive_rates, marker='.')
    plt.plot([0,1], [0,1], linestyle='--')
    plt.show()

    auc = calculate_auc(false_positive_rates, true_positive_rates)

    return auc

def calculate_auc(false_positive_rates, true_positive_rates):
    auc = 0
    true_positive_rates.sort()
    false_positive_rates.sort()
    for i in range(1, len(false_positive_rates)):
        trap_fpr = false_positive_rates[i] - false_positive_rates[i - 1]
        trap_tpr = (true_positive_rates[i] + true_positive_rates[i-1]) / float(2)

        auc += trap_fpr * trap_tpr

    return auc



def get_confusion_matrix_values(df, weights):
    TP = 0
    FP = 0
    TN = 0
    FN = 0

    for idx, row in df.iterrows():
        predicted = sigmoid(predict(row[:df.shape[1]-1], weights))
        actual = row.iloc[-1]

        if predicted > 0.5:
            predicted = 1
        else:
            predicted = 0

        if predicted == 1 and actual == 1:
            TP += 1
        elif predicted == 1 and actual == 0:
            FP += 1
        elif predicted == 0 and actual == 1:
            FN += 1
        elif predicted == 0 and actual == 0:
            TN += 1

    return {'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN}


def predict(row, weights):
    value = row.dot(weights[1:]) + weights[0]
    return value


df = pd.read_csv('/Users/justinlittman/dev/courses/DS4420/HW2/spambase.csv', header=None)

learning_rate = 0.01
epochs = 3000
num_folds = 2

run_algorithm(df, num_folds, epochs, learning_rate)