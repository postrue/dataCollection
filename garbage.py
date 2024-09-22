# takes in return from expfit function, returns 7x3x2 array where rows/cols
# are the same, slices are [R_squared, MSE] for each
def errors(amps, params):
    array_3d = np.zeros((7, 3, 2))
    x_data = np.array([0, 1, 2])
    for i in range(7):
        for j in range(3):
            y_data = amps[i, j][::-1]
            p = params[i, j]

            # Predicted y-values using the fitted model
            y_pred = exponential_func(x_data, *p)

            # Calculate the residual sum of squares (RSS)
            RSS = np.sum((y_data - y_pred)**2)

            # Calculate the total sum of squares (TSS)
            mean_y = np.mean(y_data)
            TSS = np.sum((y_data - mean_y)**2)

            # Calculate R-squared
            R_squared = 1 - (RSS / TSS)

            # Calculate Mean Squared Error (MSE)
            MSE = np.mean((y_data - y_pred)**2)
            array_3d[i, j] = [R_squared, MSE]
    return array_3d


def tester():
    x_data = np.linspace(0, 1, 100)
    # y_data = np.array([math.log(x_data[i], math.e) for i in range(1000)])
    y_data = -x_data + 1

    print("y_data: ", y_data)
    # Fit the exponential model to the data
    popt, pcov = curve_fit(exponential_func, x_data, y_data, p0 = (1,-2.6))

    # Predicted y-values using the fitted model
    y_pred = exponential_func(x_data, *popt)

    #calculate fitted curve
    # x_range = np.linspace(1, 3, 100)  # for example, range from 1 to 3 with 100 points

    # Compute corresponding y-values using the exponential function
    y_range = exponential_func(x_data, *popt)

    # Calculate the residual sum of squares (RSS)
    RSS = np.sum((y_data - y_pred)**2)

    # Calculate the total sum of squares (TSS)
    mean_y = np.mean(y_data)
    TSS = np.sum((y_data - mean_y)**2)

    # Calculate R-squared
    R_squared = 1 - (RSS / TSS)

    # Calculate Mean Squared Error (MSE)
    MSE = np.mean((y_data - y_pred)**2)
    print("R-squared: ", R_squared)
    print("Mean Squared Error (MSE): ", MSE) 

    #plot
    plt.plot(x_data, y_data)
    plt.plot(x_data, y_range, color="red")
    plt.show()


# ax = ['x','y','z']
# back = [3,2,1]
# f = open('amplitudes.txt', 'w')
# array_3d = amps('asiyah','0226')
# array_2d = decays(array_3d)
# print(array_2d)
# for t in range(7):
#     f.write(f"{t*10} Minutes:\n")
#     for i in range(3):
#         for j in range(3):
#             f.write(f"{ax[i]}{back[j]}: {array_3d[t,i,j]: .2f}\n")
#     f.write("\n")
# f.close()

def plotdecays(amps):
    ax = ['x','y','z']
    x_data = np.array([0, 1, 2])  # Assuming evenly spaced x-values
    for i in range(3):
        plt.figure()  # Create another new figure
        for j in range(7):
            y_data = amps[j, i][::-1]
            plt.subplot(1, 7, j+1)
            plt.plot(x_data, y_data)
            plt.title(f"{ax[i]} - {j*10} min")
    plt.show()


def decayovertime(decays):
    ax = ['x','y','z']
    x_data = np.arange(0, 61, 10)  # Assuming evenly spaced x-values
    plt.figure()
    for i in range(3):
        y_data = []
        for j in range(7):
            y_data.append(decays[j,i])
        plt.subplot(1,3,i+1)
        plt.plot(x_data, y_data)
        plt.title(f"{ax[i]} axis")
    plt.show()


#previously in main
def main():
    name = input("name of data to plot: ")
    date = input("date of data to plot: ")
    delim = input("enter delimeter for file: ")
    # alldecays = input("would you like to plot all the decays? (y/n): ")
    # timedecays = input("would you like to plot the decay over time? (y/n): ")
    l = ['x','y','z']
    array_3d = amps(name, date, delim)
    fit = expfit(array_3d)
    err = errors(array_3d, fit)
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
    
    # array_2d = decays(array_3d)
    # if (alldecays == 'y'):
    #     plotdecays(array_3d)
    # if (timedecays == 'y'):
    #     decayovertime(array_2d)
    # errs = errors(array_3d)
    # print(errs)


# input: takes in return value from signal_processing() function
# output: Nx3 matrix. N is the number of vibrations per reading. columns are an axis [X, Y, Z]
def individual_decays(processed_signal, N):

    ret = np.zeros((N,3))

    index_patterns = [
    [6, 3, 0],
    [7, 4, 1],
    [8, 5, 2]
    ]


    for i, indices in enumerate(index_patterns):
        ax = [processed_signal[j] for j in indices]
        len_check = np.array([len(sensor) for sensor in ax])
        if np.all(len_check == N): # if less than 30 segments detected, it was a bad signal
            for s in range(N):
                avgamps = [np.mean(np.abs(ax[j][s])) for j in range(3)]
                x_data = list(range(3))

                # Fit the exponential model to the data
                popt, pcov = curve_fit(exponential_func, x_data, avgamps)

                # Extract the parameters
                a, b = popt

                # The parameter 'b' represents the rate of growth or decay
                ret[s, i] = b
        
    return ret


#version of individual_decays that tries to work with two values:
# input: return values from signal_processing
# output DECAYS: 3xS array with each array being a direction and S being the number of vibrations in the data collection session. Will input 0 if one particular vibration was not properly captured by more than one of the three sensors.
# output AMPS: 9xS array 
def individual_decays(processed_signal, svals, lengths, tolerance = 120):
    decays = [[], [], []]  # To store decay values for each index pattern
    amps = [[],[],[],[],[],[],[],[],[]] # To store amplitudes for each sensor/direction. Order is: ['X3','Y3','Z3','X2','Y2','Z2','X1','Y1','Z1']
    index_patterns = [
        [6, 3, 0], # corresponds to [X1 X2 X3]
        [7, 4, 1], # corresponds to [Y1 Y2 Y3]
        [8, 5, 2]  # corresponds to [Z1 Z2 Z3]
    ]
    num_vals = [[[],[]],[[],[]],[[],[]]]
    # Loop through each set of indices
    for i, indices in enumerate(index_patterns):
        ax = [processed_signal[j] for j in indices]  # Extract 3D array of dimensions 3xnxs
        
        # Initialize index variables
        s1, s2, s3 = 0, 0, 0
        max_lengths = [len(ax[0]), len(ax[1]), len(ax[2])]  # Max lengths for each 2D array in ax
        

        # While loop until one of the index variables reaches the max length
        while s1 < max_lengths[0] and s2 < max_lengths[1] and s3 < max_lengths[2]:
            # Get vectors from the corresponding 2D array
            v1, v2, v3 = ax[0][s1], ax[1][s2], ax[2][s3]
            sval1, sval2, sval3 = svals[indices[0]][s1], svals[indices[1]][s2], svals[indices[2]][s3]

            within_tolerance = [abs(sval1 - sval2) <= tolerance, abs(sval2 - sval3) <= tolerance, abs(sval1 - sval3) <= tolerance]
            print(f"i: {i}, s1: {s1}, s2: {s2}, s3: {s3}, {within_tolerance}")
            print(f"sval1: {sval1}, sval2: {sval2}, sval3: {sval3}")
            # Conditional steps based on tolerance check
            if abs(sval1 - sval2) <= tolerance and abs(sval2 - sval3) <= tolerance:
                # All three vectors pass tolerance check
                amps[indices[0]].append(max(v1))
                amps[indices[1]].append(max(v2))
                amps[indices[2]].append(max(v3))
                y_values = [max(v1), max(v2), max(v3)]
                x_values = list(range(3))
                s1 += 1
                s2 += 1
                s3 += 1
                
            elif abs(sval1 - sval2) <= tolerance:
                # v1 and v2 pass tolerance check
                amps[indices[0]].append(max(v1))
                amps[indices[1]].append(max(v2))
                amps[indices[2]].append(0)
                y_values = [max(v1), max(v2)]
                x_values = list(range(2))
                s1 += 1
                s2 += 1
            elif abs(sval1 - sval3) <= tolerance:
                # v1 and v3 pass tolerance check
                amps[indices[0]].append(max(v1))
                amps[indices[1]].append(0)
                amps[indices[2]].append(max(v3))
                y_values = [max(v1), max(v3)]
                x_values = list(range(2))
                s1 += 1
                s3 += 1
                
            elif abs(sval2 - sval3) <= tolerance:
                # v2 and v3 pass tolerance check
                amps[indices[0]].append(0)
                amps[indices[1]].append(max(v2))
                amps[indices[2]].append(max(v3))
                y_values = [max(v2), max(v3)]
                x_values = list(range(2))
                s2 += 1
                s3 += 1
                
            else:
                # None pass tolerance, increment the smallest index variable and append 0
                min_index = np.argmin([v1[0], v2[0], v3[0]])
                if min_index == 0:
                    s1 += 1
                elif min_index == 1:
                    s2 += 1
                else:
                    s3 += 1
                decays[i].append(0)
                amps[indices[0]].append(0)
                amps[indices[1]].append(0)
                amps[indices[2]].append(0)
                continue  # Skip to next iteration
            
            # Perform linear approximation in logarithmic space for decay calculation
            
            log_y_values = np.log(y_values)
            slope, intercept = np.polyfit(x_values, log_y_values, 1)
            decay = -slope  # Decay rate is the negative of the slope
            decays[i].append(decay)
            num_vals[i][len(y_values) - 2].append(decay)
    
    for i, v in enumerate(num_vals):
        for j, w in enumerate(v):
            if len(w) != 0:
                if j == 0:
                    print(f"axis {i} average two-value decay: {sum(w)/len(w)}")
                if j == 1:
                    print(f"axis {i} average three-value decay: {sum(w)/len(w)}")
    return (decays, amps)

