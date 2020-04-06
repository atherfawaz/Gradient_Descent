# GRADIENT DESCENT ON IRIS/FISCHER DATASET
import pandas as pd
import numpy as np
import random


def predict_classes():
    dummy = None

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
            flower_class[idx] = 0
        elif (x == 'Versicolor'):
            flower_class[idx] = 1
        elif (x == 'Virginica'):
            flower_class[idx] = 2
        idx += 1

    data = np.random.rand(flower_class.size, 5)
    data[:, 0] = sepal_width
    data[:, 1] = sepal_length
    data[:, 2] = petal_width
    data[:, 3] = petal_length
    weight_vector = np.random.rand(5, 1)    # 4 wi for 3 features

    changes = True
    iterations = 0
    while(iterations <= 1000 and changes is True):
        predict_classes()
        iterations += 1