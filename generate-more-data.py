import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

base_signals_file = pd.read_csv('signals-1.csv', header=None)

just_signals = base_signals_file.loc[:, 0:99]
bad_signals = np.dot(just_signals, -1.0)
bad_classification = np.array([np.zeros(len(bad_signals))])
new_data = np.insert(bad_signals, 100, bad_classification, axis=1)
# Now export this data into a new csv. 
np.savetxt('signals-2.csv', new_data, delimiter=',')