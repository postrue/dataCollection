#find avg value of signal
#each data point is amplitude of one vibration
#each data chart is 3*7 x 3 (each sensor * time) x (each direction)
#3 data charts, one for x1y1z1, x2y2z2, x2y3z3
#do pca on the 3 data charts to see where amplitudes vary the most

import numpy as np
import csv
import matplotlib.pyplot as plt

user = 'asiyah'
date = '0225'

#xs = np.linspace(t0, t1, n_samples)
#ys = 7 * np.sin(15 * 2 * np.pi * xs) + 3 * np.sin(13 * 2 * np.pi * xs)
for t in range(0, 61, 10):
    file_path = f"./Data/{user}_{date}/{user}_{0}.csv"

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
                    pass 
            if len(row_values) == 10:
                for _ in range(9):
                    y_values[_].append(row_values[_])

t0 = 0
t1 = 60

ys = y_values[8]
n_samples = len(ys)
xs = np.linspace(t0, t1, n_samples)
               

plt.subplot(2, 1, 1)
plt.plot(xs, ys)

np_fft = np.fft.fft(ys)
amplitudes = 2 / n_samples * np.abs(np_fft) 
frequencies = np.fft.fftfreq(n_samples) * n_samples * 1 / (t1 - t0)

sorted_array = np.sort(amplitudes)[::-1]

# Print the 5 greatest elements
print(frequencies[:len(frequencies) // 2])
print("5 greatest elements:", sorted_array[:5])
print("max amplitudes: " + str(max(amplitudes)))
print("frequency at highest: " + str(frequencies[np.argmax(amplitudes)]))

plt.subplot(2, 1, 2)
plt.semilogx(frequencies[1:len(frequencies) // 2], amplitudes[1:len(np_fft) // 2])

plt.show()