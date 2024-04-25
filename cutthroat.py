import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


# input: file path to a csv file
# return value: 3x3 array where rows a direction, columns are a sensor
def amps(file_path):
    array_2d = np.zeros((3, 3))

    y_values = [[] for _ in range(9)]
    with open(file_path, 'r') as file:
        first_line = file.readline().strip()
        delim = ',' if ',' in first_line else ' '
        reader = csv.reader(file, delimiter=delim)            
        for row in reader:
            row_values = []
            for value in row:
                try:
                    float_value = float(value)
                    row_values.append(float_value)
                except ValueError:
                    pass 
            if len(row_values) >= 9:
                for _ in range(9):
                    y_values[_].append(row_values[_])

    for i in range(3):
        for j in range(i,9,3):
            ys = y_values[j]
            n_samples = len(ys)
            
                        
            np_fft = np.fft.fft(ys)
            amplitudes = 2 / n_samples * np.abs(np_fft) 

            array_2d[i, j//3] = max(amplitudes[100:len(np_fft) // 2])
    return array_2d


def exponential_func(x, a, b):
    return a * np.exp(b * x)


# input: return array from amps() function
# return value: A tuple consisting of two 3x2 arrays. 
# For both, rows are a direction x, y, z
# For the first array, the columns contain the parameters for an exponential function fit to 
# that data [a, b], where y = a * e^(b*x)
# For the second array, the columns contain the [R^2, MSE] errors for the corresponding fit
def expfit(amps):
    params = np.zeros((3, 2))
    errors = np.zeros((3, 2))
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    for i in range(3):
        y_data = amps[i][::-1]

        # Fit the exponential model to the data
        popt, pcov = curve_fit(exponential_func, x_data, y_data)

        params[i] = popt

        # Predicted y-values using the fitted model
        y_pred = exponential_func(x_data, *popt)

        # Calculate the residual sum of squares (RSS)
        RSS = np.sum((y_data - y_pred)**2)

        # Calculate the total sum of squares (TSS)
        mean_y = np.mean(y_data)
        TSS = np.sum((y_data - mean_y)**2)

        # Calculate R-squared
        R_squared = 1 - (RSS / TSS)

        # Calculate Mean Squared Error (MSE)
        MSE = np.mean((y_data - y_pred)**2)
        errors[i] = [R_squared, MSE]

    return (params, errors)


def main():
    directory = './Data'
    ax = ['x', 'y', 'z']
    gs = []
    axcounter = np.zeros(3)
    f = open('gooddata.txt', 'w')
    for seshname in os.listdir(directory):
        seshctr = 0
        seshpath = os.path.join(directory, seshname)
        if os.path.isdir(seshpath):
            for filename in os.listdir(seshpath):
                # Construct the full file path
                filepath = os.path.join(seshpath, filename)
                
                # Check if the current file is a regular csv file
                if os.path.isfile(filepath) and filename.endswith('.csv'):
                    
                    amplitudes = amps(filepath)
                    params, errors = expfit(amplitudes)
                    for i in range(3):
                        if errors[i, 0] > 0.9:
                            seshctr += 1
                            axcounter[i] += 1
                            f.write(f"{filepath}, {ax[i]} axis\n")
        if seshctr >= 10:
            gs.append(seshname)
    f.write(f"x: {axcounter[0]}, y: {axcounter[1]}, z:{axcounter[2]}\n")
    f.write("good sessions:\n")
    for s in gs:
        f.write(f"{s}\n")
    f.close()
                    
if __name__ == '__main__':
    main()
            
        



