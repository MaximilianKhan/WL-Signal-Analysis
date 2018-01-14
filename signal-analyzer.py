import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 

# Get the CSV file. 
signals_file = pd.read_csv('signals-1.csv', header=None)

# Observe some basic information. 
number_of_signals = len(signals_file.loc[:, 0])
size_of_signal = len(signals_file.loc[0, :])
print('Number of signals: %s' %(number_of_signals))
print('Size of signal: %s datapoints' %(size_of_signal))

# Now, analyze each signal, one by one, and classify it. 
# In the 101th column of the csv file, I will manually enter a 0 or 1. 
# A 0 will signify that the data is not valid, and the reverse for a 1. 

individual_signal_index = 2432
print('Observing signal #%s' %(individual_signal_index + 1))
# Exclude the 101th column, or 100th index, because that is where I will be storing the classification. 
individual_signal = signals_file.loc[individual_signal_index, 0:99]
x = np.linspace(0, 100, 100)

fig = plt.figure() 
ax0 = fig.add_subplot(111)
ax0.set_title('Signal #%s' %(individual_signal_index + 1))
ax0.scatter(x, individual_signal)

plt.show()