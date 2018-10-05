import os
import numpy as np
import cv2
from align import AlignDlib
from model import create_model
import pickle

alignment = AlignDlib('models/landmarks.dat')
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

knn = pickle.load(open('knn_model.sav', 'rb'))
svc = pickle.load(open('svc_model.sav', 'rb'))

def recognise(path):
    img = cv2.imread(path)
    bb = alignment.getLargestFaceBoundingBox(img)
    img_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if img_aligned is not None:
        return bb , img_aligned
    else:
        print('change pic')

def encoding(img):
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))
    return embedded

def predict(vec):
    return knn.predict(vec)


img_path = 'Recog/dab1.jpg'
bb , img_aligned = recognise(img_path)
encoded = encoding(img_aligned)
val = predict(encoded)
img1 = cv2.imread(img_path,1)
print(val)
print(type(val))
cv2.rectangle(img1,(bb.left(),bb.top()),(bb.left()+bb.width(),bb.top()+bb.height()),(0,255,0),3)
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img1,val[0],(bb.left(),bb.top()), font, 1,(255,255,255))
cv2.imshow('img', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
