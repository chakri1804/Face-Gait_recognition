import os
import pandas
import numpy as np
import cv2
from align import AlignDlib
from model import create_model

# img_path = 'images/Arnold_Schwarzenegger/Arnold_Schwarzenegger_0002.jpg'
# img_path =  'data/n000149/0391_01.jpg'
path = 'Game Faces/'
faces = os.listdir(path)
out_path = 'embedding/'

# Initialize the OpenFace face alignment utility
alignment = AlignDlib('models/landmarks.dat')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')


for face in faces:
	embeddings = []
	pics = os.listdir(os.path.join(path, face))
	for pic in pics:
		img  = cv2.imread(os.path.join(path, face, pic), 1)
		bb = alignment.getLargestFaceBoundingBox(img)
		img_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
		if img_aligned is not None:
			img_aligned = (img_aligned / 255.).astype(np.float32)
			embedding = nn4_small2_pretrained.predict(np.expand_dims(img_aligned, axis=0))
			embeddings.extend(embedding)
			print(pic)
	pandas.DataFrame(embeddings).to_csv(os.path.join(out_path, face+'.csv'), index=False, header=False, sep=',')
	print("FACE"+ face)
