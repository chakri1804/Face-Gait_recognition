import os
import cv2
import pickle
from model import create_model
from align import AlignDlib
import numpy as np

alignment = AlignDlib('models/landmarks.dat')
knn = pickle.load(open('svc_model.sav', 'rb'))
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

img = cv2.imread('Recog/ciri.jpg',1)
bb = alignment.getAllFaceBoundingBoxes(img)
font = cv2.FONT_HERSHEY_SIMPLEX
print
t = bb[0].top()
l = bb[0].left()
r = bb[0].right()
b = bb[0].bottom()

_img = img
_img = img[t:b,l:r]
_img = alignment.align(96,_img, bb[0], landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
_img = (_img / 255.).astype(np.float32)
embedded = nn4_small2_pretrained.predict(np.expand_dims(_img, axis=0))
val = knn.predict(embedded)
print val
