# GRADIENT DESCENT ON IRIS/FISCHER DATASET
import pandas as pd
import numpy as np
import random

# GET Y-HAT
def predict_classes(data, weight_vector, testing=False):
    transposed_weights = np.transpose(weight_vector)
    input_samples = data[:, : 5]
    input_samples = np.transpose(input_samples)
    pred_array = np.transpose(transposed_weights.dot(input_samples))
    if (testing):
        for i in range(0, data[:, 0].size):
            if pred_array[i] <= 1.5:
                pred_array[i] = 1
            elif pred_array[i] > 1.5 and pred_array[i] <= 2.5:
                pred_array[i] = 2
            elif pred_array[i] > 2.5:
                pred_array[i] = 3
    data[:, 6] = pred_array[:, 0]
    return (data, weight_vector)


# MAIN FUNCTION
if __name__ == "__main__":

    dataset = pd.read_excel('dataset.xlsx')

    sepal_width = np.array(dataset['Sepal width'])
    sepal_length = np.array(dataset['Sepal length'])
    petal_length = np.array(dataset['Petal length'])
    petal_width = np.array(dataset['Petal width'])
    flower_class = np.array(dataset['Species'])

    idx = 0
    for x in flower_class:
        if (x == 'Setosa'):
            flower_class[idx] = 1
        elif (x == 'Versicolor'):
            flower_class[idx] = 2
        elif (x == 'Virginica'):
            flower_class[idx] = 3
        idx += 1

    data = np.random.rand(flower_class.size, 7)
    data[:, 0] = 1
    data[:, 1] = sepal_length
    data[:, 2] = sepal_width
    data[:, 3] = petal_length
    data[:, 4] = petal_width
    data[:, 5] = flower_class
    weight_vector = np.random.rand(5, 1)
    random.shuffle(data)
    print(data)
    data_training = data[:100]
    data_testing = data[100:]

    training_rate = 0.002

    changes = True
    iterations = 0
    while(iterations <= 1000 and changes is True):
        (data_training, weight_vector) = predict_classes(
            data_training, weight_vector)
        cost = np.sum(np.abs(data_training[:, 5] - data_training[:, 6]))
        for k in range(0, weight_vector[:, 0].size):
            sumCalc = 0
            size_ = data_training[:, 0].size
            for i in range(0, data_training[:, 0].size):
                diff = np.subtract(data_training[i, 5], data_training[i, 6])
                feature = data_training[i, k]
                sumCalc += diff * feature
            weight_vector[k] = weight_vector[k] - \
                (training_rate * -1/data_training[:, 0].size * sumCalc)
        if (iterations % 100 == 0):
            print(cost)
        iterations += 1

    mismatch_count = 0
    (data_testing, weight_vector) = predict_classes(
        data_testing, weight_vector, True)
    for i in range(0, data_testing[:, 0].size):
        if data_testing[i, 5]-data_testing[i, 6] != 0:
            mismatch_count += 1

    test_cost = np.sum(np.abs(data_testing[:, 5]-data_testing[:, 6]))
    print("Mismatch count is: ", mismatch_count)
    print("Test cost is: ", test_cost)
