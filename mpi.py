#!/usr/bin/env python

from __future__ import print_function
from mpi4py import MPI
import numpy as np
import pandas as pd
import time as proctime

print ('start, ', proctime.time())

#########get parallel setting#####
comm = MPI.COMM_WORLD
size=comm.Get_size()  #number of processors
rank=comm.Get_rank()  #head processor
print ('Rank = %d, Size = %d' %(rank, size))
#################################

timestep = 5
file = 'hbonds'
df = None
colend = None
cols = None

if rank == 0:
	###########file editing############
	data = []
	numlines = 0
	numcols = []
	with open(file, 'r') as f:
	        for line in f:
	                numcols.append(len(line.split(',')))
	                numlines+=1
	f.close()
	
	coltot = np.max(numcols)
	arr = np.empty((coltot), dtype='S15')
	
	i = 0
	with open(file, 'r') as f:
	        for line in f:
	                line = np.asarray(line.strip('\n').split(','))
	                lineext = np.chararray((coltot - len(line)), itemsize=15)
	                lineext[:] = 'nan'
	                line = np.hstack((line, lineext))
	                arr = np.vstack((arr, line))
	                i+=1
	f.close()
	arr = arr[1:]
	df = pd.DataFrame(arr, index=None)
	
	if (df.iloc[0,:]=='nan').any():
	        colend = (df.iloc[0,:]=='nan').argmax()
	else:
	        colend = df.shape[1]
	##here i have complete file in a dataframe###
	#####################################

	######distributing matrix########
	cols = np.zeros((size, 2))
	num = colend/(size)
	
	for i in range(size):
		cols[i,0], cols[i,1] = (i) * num, (i+1) * num
	cols[-1,1] = colend
	cols = cols.astype(int)
	print (cols)
	print ('serial end, ', proctime.time())
	###############################

df = comm.bcast(df,root=0)
colend = comm.bcast(colend,root=0)
cols = comm.bcast(cols,root=0)

###define process in function func#####
def func(data, startcol, endcol):
        j = startcol
	time = []
        while j < endcol:
                t,i = 0,1
                while i < data.shape[0]:
                        if data.iloc[i,:].str.match(str(data.iloc[0,j])).any():
                                t+=timestep
                                i+=1
                        else:
                                break
                time.append(t)
                j+=1
	return np.asarray(time)
##################################

######run the loop distributed on ranks#########
for col in range(cols.shape[0]):
	if rank == col:
		time = func(df, cols[col,0], cols[col,1])
		np.savetxt('times' + str(rank), time, delimiter=',', fmt='%i')
################################

print ('parallel end: ', proctime.time())

