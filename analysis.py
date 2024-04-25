import numpy as np
import csv
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os


# input: file path to a csv file
# return value: 7x3x3 array where rows are a time interval, columns are a direction, 
# slices are a sensor
# example:
# [0 min:[[X3, X2, X1],[Y3,Y2,Y1],[Z3, X2, Z1]]
#  10 min:[[][][]]...
#  60 min]
def amps(user, date):
    t0 = 0
    t1 = 60
    array_3d = np.zeros((7, 3, 3))
    for t in range(0, 61, 10):
        file_path = f"./Data/{user}_{date}/{user}_{t}.csv"

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

                array_3d[t//10, i, j//3] = max(amplitudes[100:len(np_fft) // 2])
    return array_3d


def exponential_func(x, a, b):
    return a * np.exp(b * x)


# input: return array from amps() function
# return value: 7x3 array where rows are time intervals, columns are a direction, 
# and each data point is the exponential decay for that time interval/direction
# example: 
# [0 min: [Xdecay Ydecay Zdecay]
#  10 min: [] ...
#  60 min]
def decays(amps):
    array_2d = np.zeros((7, 3))
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    for i in range(7):
        for j in range(3):
            y_data = amps[i, j][::-1]

            # Fit the exponential model to the data
            popt, pcov = curve_fit(exponential_func, x_data, y_data)

            # Extract the parameters
            a, b = popt

            # The parameter 'b' represents the rate of growth or decay
            array_2d[i, j] = b

    return array_2d


# input: return array from amps() function
# return value: A tuple consisting of two 7x3x2 arrays. 
# For both, rows are time intervals and columns are a direction 
# For the first array, the slice contains the parameters for an exponential function fit to 
# that data [a, b], where y = a * e^(b*x)
# For the second array, the slice contains the [R^2, MSE] errors for the corresponding fit
def expfit(amps):
    params = np.zeros((7, 3, 2))
    errors = np.zeros((7, 3, 2))
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    for i in range(7):
        for j in range(3):
            y_data = amps[i, j][::-1]

            # Fit the exponential model to the data
            popt, pcov = curve_fit(exponential_func, x_data, y_data)

            params[i, j] = popt

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
            errors[i, j] = [R_squared, MSE]

    return (params, errors)


# input: return array from amps() function
# return value: none
# plots 3 separate figures, each with 7 subplots.
# each subplot is the decay in signal amplitude over distance for a specific direction/time
# also plots the exponential fit to the data, in red
def plotfit(amps):
    fit = expfit(amps)[0]
    ax = ['x','y','z']
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    x_range = np.linspace(0, 2, 100)
    for i in range(3):
        plt.figure()  # Create another new figure
        for j in range(7):
            y_data = amps[j, i][::-1]
            y_range = exponential_func(x_range, *fit[j, i])
            plt.subplot(2, 4, (j+1) % 8)
            plt.plot(x_data, y_data)
            plt.plot(x_range, y_range, color = "red")
            plt.title(f"{ax[i]} - {j*10} min")
    plt.show() 


def main():
    name = input("name of data to plot: ")
    date = input("date of data to plot: ")
    # alldecays = input("would you like to plot all the decays? (y/n): ")
    # timedecays = input("would you like to plot the decay over time? (y/n): ")
    l = ['x','y','z']
    array_3d = amps(name, date)
    fit, err = expfit(array_3d)
    f = open('exp_fit_info.txt', 'w')
    for ax in range(3):
        f.write(f"{l[ax]} axis:\n")
        for t in range(7):
            if t == 0:
                f.write(f"0  minutes: R_squared -{err[t, ax, 0]: .4f}, MSE -{err[t, ax, 1]: .4f}\n")
            else:
                f.write(f"{t*10} minutes: R_squared -{err[t, ax, 0]: .4f}, MSE -{err[t, ax, 1]: .4f}\n")  
        f.write("\n")
    f.close()
    plotfit(array_3d)
    
                    
if __name__ == '__main__':
    main()
            
        



