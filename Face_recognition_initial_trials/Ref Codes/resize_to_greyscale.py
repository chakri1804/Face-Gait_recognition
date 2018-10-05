import glob, os
import pandas as pd
import numpy as np
import cv2 as cv 

path = 'vggface2_train/train/'
test = open('test_names.txt', 'w')
train = open('train_names.txt', 'w')
# pics = path + pd.read_fwf('meta/train_list.txt') 
erects = pd.read_csv('bb_landmark/loose_bb_train.csv')
blah = np.asarray(erects)
# fails = []
# print(dirs)
images = []
labels = []
abn_labels = []
abn_img = []
for pic, x, y, w, h in blah:
	if(x>0 and y>0):
		print(pic)
		img = cv.imread(path+pic+'.jpg')
		img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
		img = img[y:y+h,x:x+w]
		img = cv.resize(img,(80,100))
		img_label = pic[0:7]
		cv.imwrite(path+pic+'.jpg',img)
		# images.append(img)
		labels.append(img_label)
	else:
		print(pic)
		# img = cv.imread(path+pic+'.jpg')
		img_label = pic[0:7]
		# cv.imwrite(path+pic+'.jpg',img)
		# abn_img.append(img)
		abn_labels.append(img_label)

lbls = np.array(labels)
abn_lbls = np.array(abn_labels)
np.save("label.npy",lbls)
np.save("test_labels.npy",abn_lbls)