import fun
import os
import numpy as np


# dir = 'asiyah-forearm-1'
# directory = f'./{dir}'


# fail = False
# for seshname in os.listdir(directory):
#     seshpath = os.path.join(directory, seshname)
#     if os.path.isdir(seshpath):
#         for filename in os.listdir(seshpath):
#             # Construct the full file path
#             filepath = os.path.join(seshpath, filename)
            
#             # Check if the current file is a regular csv file
#             if os.path.isfile(filepath) and filename.endswith('.csv'):
                


#                 proc, svals = fun.signal_processing(f"{dir}/{seshname}/{filename}",540,200,100)
#                 dec, amps = fun.individual_decays(proc, svals)
                
                    
#                 print(filename)
#                 for i, sub_list in enumerate(dec):
#                     non_zero_count = sum(1 for x in sub_list if x != 0)
#                     print(f"Array {i} has a length of {len(sub_list)} and {non_zero_count} nonzero values.")



proc, svals, lens = fun.signal_processing(f"asiyah-forearm-1/relax-4_0919/relax-4_1.csv",540,200,100)
dec, amps = fun.individual_decays(proc, svals, lens)
print(lens[5])
for d in dec:
    print(len(d))
# print(dec)

