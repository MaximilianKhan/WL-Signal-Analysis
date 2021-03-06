import os
from scipy.io import loadmat
from scipy.signal import hilbert, chirp
from scipy.stats import mode 
import matplotlib.pyplot as plt
import numpy as np 
import timeit

#====================================================================================
# Functions
#====================================================================================

def get_data(filename):
    data_folder_name = 'mat-data'
    # dictionary_keys = ['data', 'datastart', 'dataend', 'rangemin', 'rangemax', 'unittext',
    #             'unittextmap', 'blocktimes', 'tickrate', 'samplerate', 'firstsampleoffset',
    #             'comtext', 'com']
    mat_data = loadmat(data_folder_name + os.sep + filename)
    # We say mat_data['data'][0] because the data is stored in a two-dimensional matrix,
    # but there is only one column that contains the data that we want. 
    return mat_data['data'][0]

def get_instantaneous_frequency(data_vector):
    # Assuming that this is the sample rate per second.
    # fs is frequency of data per second
    fs = samplerate = 20000

    total_seconds = len(data_vector) / samplerate
    total_minutes = total_seconds / 60
    print('Observing file data interval of %s seconds.' %(total_seconds))

    duration = total_seconds
    samples = int(fs * duration)
    time = np.arange(samples) / fs 
    signal = data_vector 

    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
    # Now, let's abs the inst_freq to make all points of significant change valid. 
    # instantaneous_frequency = np.abs(instantaneous_frequency)

    return instantaneous_frequency, time

# This comparison can be used as a check at to whether or not two vectors are of the same size, and we can continue.
def perform_inst_comp(vector1, vector2):
    v1_length = len(vector1)
    v2_length = len(vector2)
    if v1_length != v2_length:
        print('Error. Vectors are not the same size.')
        print('Vector1 length:', vector1)
        print('Vector2 length:', vector2)
        return False
    else:
        print('Vectors same size. Continuing.')
        return True

def get_peaks_and_times(data, time):
    # Compute gradients to find where derivitives are zero.
    # I've found that np.gradient is more accurate than doing it manually. 
    gradient_of_data = np.gradient(data)

    counter = 0
    values_for_peaks = []
    times_of_peaks = []
    peak_indexes = []
    for x in gradient_of_data:
        # Computing the gradients gives values of 0.0 inaccurately. But, within a range of values close to zero, we get our desired points. 
        # Experiments reveal that finding a signal from its troph is more accurate than its peaks.
        # So, we will look below our threshhold of 50 mV for points that are close to a gradient of zero.
        if (np.abs(x) < 0.0050) and (data[counter] < -0.05):
            times_of_peaks.append(time[counter])
            values_for_peaks.append(data[counter])
            peak_indexes.append(counter)
        counter += 1

    return values_for_peaks, times_of_peaks, peak_indexes

def get_non_null_lengths_and_times(data, time):
    count = 0
    data_mode = mode(data, axis=None)[0]
    non_null_length_values = []
    times_of_lengths = []
    length_indexes = []
    for x in length_data:
        # Because the average baseline length varies from file to file, we are taking the mode value for 
        # length and having it be a lower bound, plus an observed amount that accounts for baseline length variation.
        # Also, this accounts not only for ramps, but also vibrations.
        if (x > data_mode + 0.0000025) or (x < data_mode - 0.0000025):
            non_null_length_values.append(x)
            times_of_lengths.append(time[count])
            length_indexes.append(count)
        count += 1

    return non_null_length_values, times_of_lengths, length_indexes

# Fix this function to be more efficient
def get_peaks_during_ramps(peak_indexes, length_indexes, amp_data, time):
    peaks_during_ramp = []
    good_peak_times = []
    good_peak_indexes = []
    for x in peak_indexes:
        for y in length_indexes:
            if x == y:
                peaks_during_ramp.append(amp_data[x])
                good_peak_times.append(time[x])
                good_peak_indexes.append(x)
                continue

    return peaks_during_ramp, good_peak_times, good_peak_indexes

def get_wave_interval_indices(good_peak_indexes):
    # Now, I have to use time to capture the wave using the troph peak.
    # Time step forward: 0.0030
    # Backward: 0.0020

    # Get the number of troph points that will be anlayzed -> Number of signals saved
    # Create an np array based on the number of signals found. 
    # Loop through the trophs, and save the data forward and backward of the signal in a row in the np array. 
    number_of_signals = len(good_peak_indexes)
    # The first column will contain the starting index, and the second will be the ending index. 
    signal_wave_intervals = np.empty([number_of_signals, 2])

    counter = 0
    for i in good_peak_indexes:
        # So know we know the index for where the peak occured. 
        # Now, find the indexes for where the start and end times should be 

        # This is the samplerate per second
        samplerate = 20000
        subtract_num = samplerate * 0.0030
        add_num = samplerate * 0.0020

        starting_index = i - int(subtract_num)
        ending_index = i + int(add_num)

        signal_wave_intervals[counter][0] = starting_index
        signal_wave_intervals[counter][1] = ending_index

        counter += 1

    return signal_wave_intervals

def get_possible_signals(amp_data, time, signal_wave_intervals):
    # Now, go and get the data for where the signals exist using signal_value_intervals. 
    signals_found = []
    time_of_signals_found = []
    for signal in signal_wave_intervals:
        starting_index = int(signal[0])
        ending_index = int(signal[1])
        # print('Starting: %s • Ending: %s' %(starting_index, ending_index))

        signals_found.append(amp_data[starting_index:ending_index])
        time_of_signals_found.append(time[starting_index:ending_index])

    signals_found = np.array(signals_found)
    time_of_signals_found = np.array(time_of_signals_found)

    return signals_found, time_of_signals_found

#====================================================================================
# Main
#====================================================================================

UPPER_BOUND = 800000

print('Starting analysis.')
start = timeit.default_timer()

# Make sure that labchart exports everything with a constant sampling rate. 
amp_data = get_data('amp-data-2.mat')
length_data = get_data('length-data-2.mat')

# For testing purposes, we are limiting the size of our used data to accomodate for time of calculations. 
amp_data = amp_data[:UPPER_BOUND]
length_data = length_data[:UPPER_BOUND]

# Get the instantaneous frequency of our signal, and an array containing a series of our time for which the data occurs over. 
# I will most likely have to get the indexes for the inst_freq peaks as well...
inst_freq, time = get_instantaneous_frequency(amp_data)
print('(1/7) Calculated instantaneous frequency.')

# Get the peaks and the times at which they occur.
values_for_peaks, times_of_peaks, peak_indexes = get_peaks_and_times(amp_data, time)
print('(2/7) Calculated peaks.')



# Before I continue here, and add extra computational cost, I should clean the value_for_peak array. 
# We cannot have peak values too close together. Perhaps this is where frequency analysis comes into play. 



# Find where the ramps and event markers occur. 
# Maybe later use clustering to differentiate between the event markers and ramps. 
non_null_length_values, times_of_lengths, length_indexes = get_non_null_lengths_and_times(length_data, time)
print('(3/7) Calculated ramps and vibrations.')

# Find the peaks in range of the ramps. 
peaks_during_ramp, good_peak_times, good_peak_indexes = get_peaks_during_ramps(peak_indexes, length_indexes, amp_data, time)
print('(4/7) Calculated peaks during ramps and vibrations.')

# Find the indice interval of the signals during the ramp. 
signal_wave_intervals = get_wave_interval_indices(good_peak_indexes)
print('(5/7) Calculated signal intervals.')

# Now, go and get the data for where the signals exist using  signal_value_intervals. 
signals_found, time_of_signals_found = get_possible_signals(amp_data, time, signal_wave_intervals)
print('(6/7) Calculated possible signals.')


# Now with getting our signals during the ramp working, I need to disciminate which are signals and which are not. 

# The length of each signal vector is 100 indices.
# np.savetxt('signals-1.csv', signals_found, delimiter=',')
print('(7/7) SKIPPING: Saved possible signals.')



# Now that I have hand classified wave forms, I need to train a machine learning application to properly 
# differentiate between what is and is not a signal. 
# I will creating a training and testing program in separate files, save the weights and biases that are 
# accurate in some weights file (I believe that you can save tensorboard models and load them).
# After I have effectively created a working model, I will import it here to evaluate my possible signals. 


# In the end of my program, I will spit out the desired that the lab is looking for, using the data that I 
# calculated. 


# Finished analysis
stop = timeit.default_timer()
print('Analysis complete. Took %s seconds.' %(stop - start))

print('Plotting data.') 
fig = plt.figure()
# Signal
ax0 = fig.add_subplot(411)
ax0.set_title('Signal')
ax0.plot(time[:], amp_data)
ax0.plot(times_of_peaks, values_for_peaks, 'ro')
ax0.plot(good_peak_times, peaks_during_ramp, 'go')

# Length
ax1 = fig.add_subplot(412)
ax1.set_title('Length')
ax1.plot(time[:], length_data)
ax1.plot(times_of_lengths, non_null_length_values, 'ro')

# Instantaneous Frequency
ax2 = fig.add_subplot(413)
ax2.set_title('Instantaneous Frequency')
ax2.plot(time[1:], inst_freq)

# Found signals
ax3 = fig.add_subplot(414)
ax3.set_title('Signals Found')
ax3.scatter(time_of_signals_found.ravel(), signals_found.ravel())

plt.show()