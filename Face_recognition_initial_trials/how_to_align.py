import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from align import AlignDlib

# img_path = 'images/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0002.jpg'
# img_path =  'data/n000149/0391_01.jpg'
path = 'vggface2_test/test'
faces = os.listdir(path)
out_path = 'vggface2_test/aligned/'



# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')


for face in faces:
	os.system('mkdir ' + out_path+face)
	pics = os.listdir(path+face)
	for pic in pics:
		img = cv2.imread(path+face+'/'+pic, 1)
		bb = alignment.getLargestFaceBoundingBox(img)
		# img_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
		if img_aligned is not None:
			cv2.imshow('img',img)
			cv2.
			# cv2.imwrite(out_path+face+'/'+pic, img_aligned)
		print(pic)
