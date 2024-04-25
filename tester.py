import numpy as np
from scipy.optimize import curve_fit

# Define the exponential function
def exponential_func(x, a, b):
    return a * np.exp(b * x)

# Given data
x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
y_data = np.array([1, 2.7, 7.3])  # Fit the exponential model to the data
popt, pcov = curve_fit(exponential_func, x_data, y_data)

# Extract the parameters
a, b = popt

# The parameter 'b' represents the rate of growth or decay
rate_of_growth_or_decay = b
print("Rate of growth or decay:", rate_of_growth_or_decay)