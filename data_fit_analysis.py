import fun
import numpy as np
import os

def main():
    dir = input("name of directory containing sessions: ")
    thresh = int(input("threshold for number of decay models that fit well: ")) # max is num intervals * 3, good threshold is half this value
    directory = f'./{dir}'
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
                    
                    amplitudes = fun.amps_fp(filepath)
                    params, errors = fun.expfit_fp(amplitudes)
                    for i in range(3):
                        if errors[i, 0] > 0.9:
                            seshctr += 1
                            axcounter[i] += 1
                            f.write(f"{filepath}, {ax[i]} axis\n")
        if seshctr >= thresh:
            gs.append(seshname)
    f.write(f"x: {axcounter[0]}, y: {axcounter[1]}, z:{axcounter[2]}\n")
    f.write("good sessions:\n")
    for s in gs:
        f.write(f"{s}\n")
    f.close()
                    
if __name__ == '__main__':
    main()
            
        



