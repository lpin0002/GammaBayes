import os, time, sys, numpy as np
from matplotlib import colormaps as cm

from tqdm import tqdm
from zipfile import ZipFile

try:
    identifier = sys.argv[1]
except:
    identifier = time.strftime("%d%m%H")
    
    
currentdirecyory = os.getcwd()
stemdirectory = currentdirecyory+f'/data/{identifier}'
print("\nstem directory: ", stemdirectory, '\n')

rundirs = [x[0] for x in os.walk(stemdirectory)][1:]
print("number of run directories: ", len(rundirs), '\n')


for rundir in tqdm(rundirs, total=len(rundirs)):
    bkgmargresults       = np.load(f'{rundir}/bkgmargresults.npy', allow_pickle=True)
    propmargresults      = np.load(f'{rundir}/propmargresults.npy', allow_pickle=True)
    np.savez_compressed(f'{rundir}/bkgmargresults.npz', bkgmargresults = bkgmargresults)
    np.savez_compressed(f'{rundir}/propmargresults.npz', bkgmargresults = bkgmargresults)
    
