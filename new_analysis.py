import fun

proc = fun.signal_processing("asiyah-back-2/sitstretch_0710/sitstretch_9.csv",540,200,100)
dec = fun.individual_decays(proc,30)

print(dec)