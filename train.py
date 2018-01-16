import os
import numpy as np 
import pandas as pd 

# Import all of our data contained in the csv
training_data = pd.read_csv('training-data' + os.sep + 'training-data.csv', header=None)
X_train = training_data.loc[:, 0:99]
y_train = training_data.loc[:, 100]

# I need to also get test data...