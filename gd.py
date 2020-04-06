# GRADIENT DESCENT ON IRIS/FISCHER DATASET
import pandas as pd
import numpy as np

# MAIN FUNCTION
if __name__ == "__main__":

    dataset = pd.read_excel('dataset.xlsx')
    sepal_width = dataset['Sepal width']
    sepal_length = dataset['Sepal length']
    petal_length = dataset['Petal length']
    flower_class = dataset['Species']

    idx = 0
    for x in flower_class:
        if (x == 'Setosa'):
            flower_class[idx] = 0
        elif (x == 'Versicolor'):
            flower_class[idx] = 1
        elif (x == 'Virginica'):
            flower_class[idx] = 2
        idx += 1
