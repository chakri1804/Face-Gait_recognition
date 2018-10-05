import glob, os
import pandas as pd
import numpy as np
import cv2 as cv 

path = 'vggface2_test/train/'
out_path = 'vggface2_train/test/'
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
	if(x<0 or y<0):
		os.system('mv '+ path+pic+'.jpg '+out_path)
		print 'moving '+ path+pic+'.jpg to '+out_path