import fun

def main():
    fname = input("name of session folder (include directory): ")
    sl, il, vr = fun.seshinfo(fname)
    n = (sl // il) + 1
    l = ['x','y','z']
    array_3d = fun.amps(fname)
    fit, err = fun.expfit(array_3d)
    f = open('exp_fit_info.txt', 'w')
    for ax in range(3):
        f.write(f"{l[ax]} axis:\n")
        for t in range(n):
            f.write(f"{t*il} minutes: R_squared -{err[t, ax, 0]: .4f}, MSE -{err[t, ax, 1]: .4f}\n")  
        f.write("\n")
    f.close()
    fun.plotfit(array_3d)
    
                    
if __name__ == '__main__':
    main()
            
        



