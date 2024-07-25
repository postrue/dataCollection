import fun
import numpy as np
import os

def main():
    dir = input("name of directory containing sessions: ")
    directory = f'./{dir}'
    ax = ['x', 'y', 'z']
    gs = []
    axcounter = np.zeros(3)
    f = open('goodslopedata.txt', 'w')
    for seshname in os.listdir(directory):
        seshgood = False
        seshpath = os.path.join(directory, seshname)
        if os.path.isdir(seshpath):
            array_3d = fun.amps(f'{dir}/{seshname}')
            fit, exp_err = fun.expfit(array_3d)
            slopes, lin_err = fun.calcdecays(fit)
            for i in range(3):
                if lin_err[i] > 0.6:
                    seshgood = True
                    axcounter[i] += 1
                    f.write(f"{seshname}, {ax[i]} axis\n")
                    f.write(f"\terror:{lin_err[i]}, slope:{slopes[i]}\n")
        if seshgood:
            gs.append(seshname)
    f.write(f"x: {axcounter[0]}, y: {axcounter[1]}, z:{axcounter[2]}\n")
    f.write("good sessions:\n")
    for s in gs:
        f.write(f"{s}\n")
    f.close()
                    
if __name__ == '__main__':
    main()