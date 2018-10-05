import cv2
import numpy as np
import os
import pickle
from model import create_model
from align import AlignDlib

alignment = AlignDlib('models/landmarks.dat')
knn = pickle.load(open('svc_model.sav', 'rb'))
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

img = cv2.imread('Recog/lul.jpg',1)
bb = alignment.getAllFaceBoundingBoxes(img)
font = cv2.FONT_HERSHEY_SIMPLEX

# encodes = []
# labels = []

# for a in bb:
#     t = a.top()
#     l = a.left()
#     r = a.right()
#     b = a.bottom()
#     print t,l,b,r
#     _img = img
#     _img = img[t:b,l:r]
#     _img = alignment.align(96,_img, a, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
#     _img = (_img / 255.).astype(np.float32)
#     embedded = nn4_small2_pretrained.predict(np.expand_dims(_img, axis=0))
#     encodes.append(embedded)

# print encodes

# for a in range(len(bb)):
#     val = knn.predict(encodes[a])
#     labels.append(val)
#
# print val

for a in bb:
    t = a.top()
    l = a.left()
    r = a.right()
    b = a.bottom()
    # _img = img
    _img = img[t:b,l:r]
    _img = alignment.align(96,_img)
    cv2.imshow('img',_img)
    _img = (_img / 255.).astype(np.float32)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    embedded = nn4_small2_pretrained.predict(np.expand_dims(_img, axis=0))
    print embedded
    val = knn.predict(embedded)
    print val
    cv2.rectangle(img,(l,t),(r,b),(0,255,0),3)
    cv2.putText(img,val[0],(l,t), font, 1,(255,255,255))

cv2.imshow('img',img)
cv2.imwrite('res1.jpg',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
