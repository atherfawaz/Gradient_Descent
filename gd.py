#GRADIENT DESCENT ON IRIS/FISCHER DATASET

import pandas as pd
import numpy as np

dataset = pd.read_excel('dataset.xlsx')
sepal_width = dataset['Sepal width']
sepal_length = dataset['Sepal length']
petal_length = dataset['Petal length']