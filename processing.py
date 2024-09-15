import numpy as np
import csv
from scipy.signal import butter, filtfilt
import matplotlib.pyplot as plt



fs = 540

# Butterworth filter design
def butter_filter(signal, cutoff_freq, fs, order=5, filter_type='low'):
    nyquist = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff_freq / nyquist  # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype=filter_type, analog=False)
    filtered_signal = filtfilt(b, a, signal)  # Apply the filter
    return filtered_signal

# Low-pass and high-pass filter cutoff frequencies
low_cutoff = 200  # Low-pass filter cutoff frequency (Hz)
high_cutoff = 100  # High-pass filter cutoff frequency (Hz)


# Step 1: Read the CSV file
# for t in range(10):
    # file_path = f"./asiyah-back-2/{user}_{date}/{user}_{t}.csv"
file_path = "./asiyah-back-2/sitstretch_0710/sitstretch_6.csv"

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


# Step 3: Plot the extracted values
# coords = [[0,0],[0,1],[0,2],[1,0],[1,1],[1,2],[2,0],[2,1],[2,2]]
# titles = ['X3','Y3','Z3','X2','Y2','Z2','X1','Y1','Z1']
# fig, axs = plt.subplots(3, 3)
# for i in range(len(coords)):

signal = y_values[2][fs:] #hardcoding for good data
# for i in range(9):
#     signal = y_values[i][fs:] #hardcoding for good data

#     # Remove DC component by subtracting the mean
#     signal_detrended = signal - np.mean(signal)

#     # Compute the FFT of the detrended signal
#     N = len(signal_detrended)  # Number of samples
#     fft_values = np.fft.fft(signal_detrended)
#     fft_freqs = np.fft.fftfreq(N, d=1/fs)  # Frequency axis

#     # Compute the magnitude of the FFT (absolute value)
#     fft_magnitude = np.abs(fft_values)

#     # Plot the FFT result
#     plt.figure(figsize=(12, 6))
#     plt.plot(fft_freqs[:N//2], fft_magnitude[:N//2])  # Plot only the positive frequencies
#     plt.title(f"Frequency Spectrum of the Signal (DC Component Removed): channel {i}")
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Magnitude')
#     plt.grid(True)
#     plt.show()

# Step 1: Apply a high-pass filter to remove low-frequency drift
high_passed_signal = butter_filter(signal, high_cutoff, fs, order=3, filter_type='high')

# Step 2: Apply a low-pass filter to remove high-frequency noise
filtered_signal = butter_filter(high_passed_signal, low_cutoff, fs, order=3, filter_type='low')


# Threshold-based segmentation (with your adjusted threshold calculation)
threshold = np.mean(filtered_signal) + np.std(filtered_signal) * 0.5  # Adjust based on your signal
above_threshold = np.where(filtered_signal > threshold)[0]
segments = np.split(above_threshold, np.where(np.diff(above_threshold) > 100)[0] + 1) #adjust value of 100 based on signal
print(len(segments))

# Plot the original signal
plt.figure(figsize=(10, 6))
plt.plot(filtered_signal, label='Filtered Signal', alpha=0.5)

# Loop through each segment and plot only the middle 50%
for segment in segments:
    if len(segment) > 10:  # Skip short segments that may be noise
        start, end = segment[0], segment[-1]
        segment_length = end - start + 1
        
        # Keep only the middle 50% of the segment
        middle_start = start + int(0.25 * segment_length)
        middle_end = start + int(0.75 * segment_length)
        
        plt.plot(range(middle_start, middle_end+1), filtered_signal[middle_start:middle_end+1], label=f'Vibration {middle_start}-{middle_end}')

# Customize and show plot
plt.axhline(threshold, color='red', linestyle='--', label='Threshold')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.title('Middle 50% of Each Vibration')
plt.show()
