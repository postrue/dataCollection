import csv
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt


"""
Data processing steps:
    - first filter out all frequencies above 200 Hz or below 100 Hz. Main peak occurs around 160 Hz
    - then set a threshold to be the average of the signal plus half a standard deviation
    - find all indices where the signal is above the threshold
    - find the "gap threshold" this is where the distance between indices found above is in the 99th percentile of all distances between indices
    - split into segments wherever the distance between indices is greater than the gap threshold
    - skip segments with a length of less than 50 indices
    
"""

user = 'relax-3'
date = '0919'
fs = 540

# Low-pass and high-pass filter cutoff frequencies
low_cutoff = 200  # Low-pass filter cutoff frequency (Hz)
high_cutoff = 100  # High-pass filter cutoff frequency (Hz)

# Butterworth filter design
def butter_filter(signal, cutoff_freq, fs, order=5, filter_type='low'):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_signal = filtfilt(b, a, signal)  # Apply the filter
    return filtered_signal



# Step 1: Read the CSV file
for t in range(2):
    file_path = f"./asiyah-forearm-1/{user}_{date}/{user}_{t}.csv"

    # Step 2: Parse the file and extract every third value starting from the first
    y_values = [[] for _ in range(9)]
    with open(file_path, 'r') as file:
        reader = csv.reader(file, delimiter=' ')
        ondetect = False
        for row in reader:
            row_values = []
            for value in row:
                try:
                    float_value = float(value)
                    row_values.append(float_value)
                except ValueError:
                    pass  # Skip non-numeric values
                
            # row_values = [float(value) for value in row]
            # extracted_values = row_values[2::3]  # Extract every third value starting from the first
            if len(row_values) == 10:
                for _ in range(9):
                    y_values[_].append(row_values[_])
                # extracted_values = row_values[0]  # Extract every third value starting from the first
                # y_values.append(extracted_values)
    seg_lengths = []
    # Step 3: Plot the extracted values
    coords = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
    titles = ['X3','Y3','Z3','X2','Y2','Z2','X1','Y1','Z1']
    fig, axs = plt.subplots(3, 3, figsize=(10, 6))
    for i in range(len(coords)):
        signal = y_values[i][fs:]

        # Step 1: Apply a high-pass filter to remove low-frequency drift
        high_passed_signal = butter_filter(signal, high_cutoff, fs, order=3, filter_type='high')

        # Step 2: Apply a low-pass filter to remove high-frequency noise
        filtered_signal = butter_filter(high_passed_signal, low_cutoff, fs, order=3, filter_type='low')

        # Threshold-based segmentation (with your adjusted threshold calculation)
        threshold = np.mean(filtered_signal) + np.std(filtered_signal) * 0.5  # Adjust based on your signal
        above_threshold = np.where(filtered_signal > threshold)[0]

        diffs = np.diff(above_threshold)
        gap_threshold = np.percentile(diffs, 99)  # You can adjust the percentile as needed

        segments = np.split(above_threshold, np.where(diffs > gap_threshold)[0] + 1) 

    
        seg_lengths.append(len(segments)) 

        ax = axs[coords[i][0], coords[i][1]]
        ax.plot(filtered_signal, alpha=0.5)
        ax.set_xlabel('Index')
        ax.set_ylabel('Signal')
        ax.set_title(f"{titles[i]} - {t} min")
        ax.axhline(threshold, color='red', linestyle='--', label='Threshold')

        
        actual = 0
        # Loop through each segment and plot only the middle 50%
        for segment in segments:
            if len(segment) > 50:  # Skip short segments that may be noise
                actual += 1
                start, end = segment[0], segment[-1]
                segment_length = end - start + 1
                
                # Keep only the middle 50% of the segment
                middle_start = start + int(0.25 * segment_length)
                middle_end = start + int(0.75 * segment_length)
                
                ax.plot(range(middle_start, middle_end+1), filtered_signal[middle_start:middle_end+1], label=f'Vibration {middle_start}-{middle_end}')
        # print(f"{titles[i]}: {actual} segments")
    #axs[0,0].grid(True)
    # print(seg_lengths)
    plt.tight_layout()
    plt.show()


# try:
#     float_value = float(value)
#     row_values.append(float_value)
# except ValueError:
#     pass  # Skip non-numeric values