import os, sys
rundir = sys.argv[1]
dirthingy = os.getcwd()
currentdir = dirthingy+f'/data/{rundir}'
print(currentdir)

dirs = [x[0] for x in os.walk(currentdir)][1:]

print(dirs)