import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from scipy.optimize import curve_fit
from sklearn.linear_model import LinearRegression
from pathlib import Path
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import os

#input: string that is the name of the folder containing the session. ex: "Data/asiyah_0225".
#output: tuple of two integers, first is session length and second is interval length (minutes)
def seshinfo(folder):
    file_path = Path(f"./{folder}/session_info.txt")
    if file_path.exists():
        with open(file_path, 'r') as file:
            for _ in range(4):
                line = file.readline().strip().split(' ')
            sl = int(line[2])
            line = file.readline().strip().split(' ')
            il = int(line[2])
    else:
        sl = 60
        il = 10
    return (sl, il)


# input: string that is the name of the folder containing the session. ex: "Data/asiyah_0225".
# return value: nx3x3 array where rows are a time interval, columns are a direction, 
# slices are a sensor. 
# example:
# [0 min:[[X3, X2, X1],[Y3,Y2,Y1],[Z3, Z2, Z1]]
#  10 min:[[][][]]...
#  60 min]
def amps(fname):
    sl, il = seshinfo(fname)
    n = (sl // il) + 1
    array_3d = np.zeros((n, 3, 3))
    paths = fname.split('/')
    dir = paths[0]
    parts = paths[1].split('_')
    name = parts[0]
    date = parts[1]
    for t in range(0, sl + 1, il):
        file_path = f"./{dir}/{name}_{date}/{name}_{t}.csv"
        y_values = [[] for _ in range(9)] # row 1 has all the X3, row 2 all the X2, etc. each column is a data point
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
            for j in range(i,9,3): # by doing this, we focus in on one sensor at a time
                ys = y_values[j]
                n_samples = len(ys)
                
                            
                np_fft = np.fft.fft(ys)
                amplitudes = 2 / n_samples * np.abs(np_fft) 

                array_3d[t//il, i, j//3] = max(amplitudes[100:len(np_fft) // 2]) # array_3d[time, sensor, direction]
    return array_3d


# input: file path to csv file to analyze.
# return value: 3x3 array where rows are a direction and columns are a sensor. 
def amps_fp(file_path):
    
    array_2d = np.zeros((3, 3))

    y_values = [[] for _ in range(9)] # row 1 has all the X3, row 2 all the X2, etc. each column is a data point
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
        for j in range(i,9,3): # by doing this, we focus in on one sensor at a time
            ys = y_values[j]
            n_samples = len(ys)
            
                        
            np_fft = np.fft.fft(ys)
            amplitudes = 2 / n_samples * np.abs(np_fft) 

            array_2d[i, j//3] = max(amplitudes[100:len(np_fft) // 2]) # array_2d[sensor, direction]
    return array_2d


def exponential_func(x, a, b):
    return a * np.exp(b * x)

# input: return array from amps() function
# return value: nx3 array where rows are time intervals, columns are a direction, 
# and each data point is the exponential decay for that time interval/direction
# example: 
# [0 min: [Xdecay Ydecay Zdecay]
#  10 min: [] ...
#  60 min]
def decays(amps):
    n = amps.shape[0]
    array_2d = np.zeros((n, 3))
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    for i in range(n):
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
# return value: A tuple consisting of two nx3x2 arrays. 
# For both, rows are time intervals and columns are a direction 
# For the first array, the slice contains the parameters for an exponential function fit to 
# that data [a, b], where y = a * e^(b*x)
# For the second array, the slice contains the [R^2, MSE] errors for the corresponding fit
def expfit(amps):
    n = amps.shape[0]
    params = np.zeros((n, 3, 2))
    errors = np.zeros((n, 3, 2))
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    for i in range(n):
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


# input: return array from amps_fp() function
# return value: A tuple consisting of two 3x2 arrays. 
# For both, rows are directions
# For the first array, the columns represent the parameters for an exponential function fit to 
# that data [a, b], where y = a * e^(b*x)
# For the second array, the columns represent the [R^2, MSE] errors for the corresponding fit
def expfit_fp(amps):
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


# input: return array from amps() function
# return value: none
# plots 3 separate figures, each with n subplots.
# each subplot is the decay in signal amplitude over distance for a specific direction/time
# also plots the exponential fit to the data, in red
def plotfit(amps):
    n = amps.shape[0]
    fit = expfit(amps)[0]
    ax = ['x','y','z']
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    x_range = np.linspace(0, 2, 100)
    if n > 2:
        sz = next_non_prime(n)
    else:
        sz = n
    d1, d2 = closest_factors(sz)
    for i in range(3):
        plt.figure()  # Create another new figure
        for j in range(n):
            y_data = amps[j, i][::-1]
            y_range = exponential_func(x_range, *fit[j, i])
            plt.subplot(d1, d2, j+1)
            plt.plot(x_data, y_data)
            plt.plot(x_range, y_range, color = "red")
            plt.title(f"{ax[i]} - interval {j}")
    plt.show() 
    

# input: first return array from expfit() function
# return value: two 3-value vectors. first has the slopes of the predicted lines
# second has the R^2 errors for x, y, and z
# note: this only works for hour long sessions where data is collected at 10 min intervals
def calcdecays(params):
    errors = np.zeros((3,1))
    slopes = np.zeros((3,1))
    model = LinearRegression()
    num_ints = params.shape[0]
    dec = np.zeros((3, num_ints))
    ax = ['x','y','z']
    x = np.array(range(num_ints)).reshape(-1, 1)
    for i in range(num_ints):
        for j in range(3):
            dec[j,i] = params[i,j,1]
    for i in range(3):
        y = dec[i]
        model.fit(x, y)
        slopes[i] = model.coef_[0]
        y_pred = model.predict(x)
        errors[i] = r2_score(y, y_pred)
    return slopes, errors


# input: first return array from expfit() function
# return value: none
# plots a figure with three subplots for each direction
# each graph is the change in the rate of decay over an hour time period
# note: this only works for hour long sessions where data is collected at 10 min intervals

def plotdecays(params):
    
    model = LinearRegression()
    num_ints = params.shape[0]
    dec = np.zeros((3, num_ints))
    ax = ['x','y','z']
    x = np.array(range(num_ints)).reshape(-1, 1)
    for i in range(num_ints):
        for j in range(3):
            dec[j,i] = params[i,j,1]
    plt.figure()
    for i in range(3):
        y = dec[i]
        model.fit(x, y)
        x_fit = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
        y_fit = model.predict(x_fit)
        y_pred = model.predict(x)
        
        plt.subplot(1, 3, i+1)
        plt.plot(x, y)
        plt.plot(x_fit, y_fit, color='red')
        plt.title(f"{ax[i]}")
    plt.show() 


def closest_factors(n):
    # Start checking from the square root of n and go downwards
    root = int(math.sqrt(n))
    
    for i in range(root, 0, -1):
        if n % i == 0:  # If i is a factor
            return (i, n // i) # smaller first
    
    return None  # In case no factors are found, though this shouldn't happen for positive integers

def is_prime(n):
    """Check if a number is prime."""
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def next_non_prime(n):
    """Return the closest greater number that is not prime if n is prime. Otherwise, return n."""
    if not is_prime(n):
        return n
    candidate = n + 1
    while is_prime(candidate):
        candidate += 1
    return candidate

        




