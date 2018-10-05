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
	os.system('mkdir ' + out_path+face)
	pics = os.listdir(path+face)
	for pic in pics:
		img = cv2.imread(path+face+'/'+pic, 1)
		bb = alignment.getLargestFaceBoundingBox(img)
		# img_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
		if img_aligned is not None:
			cv2.imwrite(out_path+face+'/'+pic, img_aligned)
			cv2.rectangle(img1,(bb.left(),bb.top()),(bb.right(),bb.bottom()),(0,255,0),3)
			cv2.waitKey(0)
			cv2.destroyAllWindows()
		print(pic)
