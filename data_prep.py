import fun
import numpy as np

dir = 'asiyah-forearm-1'
directory = f'./{dir}'


def process_alternating(seshname, fs, low_cutoff, high_cutoff, even = False):

    proc, svals, lengths = fun.signal_processing(seshname, fs, low_cutoff, high_cutoff)
    dec, amps = fun.individual_decays(proc, svals, lengths)


    res = np.zeros((15, 12))

    for i in range(30):
        if i%2 == int(even):
            continue
        else:
            for j in range(12):
                new_ind = int((i - 1) // 2)
                if j < 3:
                    res[new_ind, j] = dec[j][new_ind]
                else:
                    res[new_ind, j] = amps[j-3][new_ind]
    return res

def process_simple(seshname, fs, low_cutoff, high_cutoff):

    proc, svals, lengths = fun.signal_processing(seshname, fs, low_cutoff, high_cutoff)
    dec, amps = fun.individual_decays(proc, svals, lengths)

    res = np.zeros((30,12))
    for i in range(30):
        for j in range(12):
            if j < 3:
                res[i, j] = dec[j][i]
            else:
                res[i, j] = amps[j-3][i] 
            
    return res

flex_3_0 = process_alternating(f"{dir}/flex-3_0919/flex-3_0.csv", 540, 200, 100)
flex_3_1 = process_alternating(f"{dir}/flex-3_0919/flex-3_1.csv", 540, 200, 100)
flex_4_0 = process_alternating(f"{dir}/flex-4_0919/flex-4_0.csv", 540, 200, 100)
flex_4_1 = process_alternating(f"{dir}/flex-4_0919/flex-4_1.csv", 540, 200, 100)

flex_data = np.vstack((flex_3_0, flex_3_1, flex_4_0, flex_4_1))
np.save('flex_data.npy', flex_data)

relax_3_0 = process_simple(f"{dir}/relax-3_0919/relax-3_0.csv", 540, 200, 100)
relax_3_1 = process_simple(f"{dir}/relax-3_0919/relax-3_1.csv", 540, 200, 100)

relax_data = np.vstack((relax_3_0, relax_3_1))
np.save('relax_data.npy', relax_data) 



                
                