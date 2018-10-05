from model import create_model
import numpy as np
import cv2
import os
import pandas

path = 'aligned/'
faces = os.listdir(path)
out_path = 'embeddings/'

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')


for face in faces:
	embeddings = []
	pics = os.listdir(os.path.join(path, face))
	for pic in pics:
		img  = cv2.imread(os.path.join(path, face, pic), 1)
		img = (img / 255.).astype(np.float32)
		embedding = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))
		embeddings.extend(embedding)
	pandas.DataFrame(embeddings).to_csv(os.path.join(out_path+face+'.csv'), index=False, header=False, sep=',')
	print(face)
