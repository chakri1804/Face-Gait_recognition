import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import os
from align import AlignDlib

# img_path = 'images/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0002.jpg'
# img_path =  'data/n000149/0391_01.jpg'
path = 'lul_dataset'
faces = os.listdir(path)
out_path = 'aligned_pics/'



# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')


for face in faces:
	# os.system('mkdir ' + out_path+face)
	# pics = os.listdir(path+face)
	img = cv2.imread(path+'/'+face, 1)
	bb = alignment.getLargestFaceBoundingBox(img)
	cv2.rectangle(img,(bb.left(),bb.top()),(bb.right(),bb.bottom()),(0,255,0),3)
	cv2.imshow('img', img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()
	print(face)
	# img_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
	# for pic in pics:
