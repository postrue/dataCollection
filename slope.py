import fun


def main():
    fname = input("name of session folder (include directory): ")
    array_3d = fun.amps(fname)
    fit, exp_err = fun.expfit(array_3d)
    fun.plotdecays(fit)    
           
if __name__ == '__main__':
    main()