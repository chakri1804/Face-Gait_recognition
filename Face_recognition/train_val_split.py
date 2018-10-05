import os
source1 = "aligned/"
val1 = "val"
files = os.listdir(source1)
import shutil
import numpy as np

for f in files:
	print f
	for p in os.listdir('aligned/'+f):
		if np.random.rand(1) < 0.2:
			os.system('mkdir val/'+f)
			os.system('mv aligned/'+f+'/'+p + ' ' + 'val/'+f+'/')

# files_valid = os.listdir(val1)
# for f in files_valid:
# 	p = os.listdir(f)
# 	for phot in p:
# 		os.system('mkdir '+ source1 + f)
# 		if np.random.rand(1) > 0.2:
# 			os.system('mv '+ val1 + f + '/' + phot + ' ' + source1 + f + '/')
