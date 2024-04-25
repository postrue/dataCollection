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