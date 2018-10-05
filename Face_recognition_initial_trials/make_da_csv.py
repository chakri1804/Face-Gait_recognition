import glob, os
import pandas as pd
import numpy as np
import cv2 as cv

path = 'vggface2_train/train/'
bad = 'vggface2_train/bad/'
# os.system('mkdir '+bad)

erects = pd.read_csv('bb_landmark/loose_bb_train.csv')
blah = np.asarray(erects)

for pic, x, y, w, h in blah:
	if(x>0 and y>0):
		print(pic)
		img = cv.imread(path+pic+'.jpg')
		# img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		img = img[y:y+h,x:x+w]
		img = cv.resize(img,(80,100))
		cv.imwrite(path+pic+'.jpg', img)
	else:
		os.system('mkdir '+bad+pic[0:8])
		os.system('mv '+path+pic+'.jpg '+bad+pic+'.jpg')

# lbls = np.array(labels)
# abn_lbls = np.array(abn_labels)
# np.save("label.npy",lbls)
# np.save("test_labels.npy",abn_lbls)
