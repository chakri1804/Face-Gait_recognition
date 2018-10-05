import cv2
import numpy as np
import os
import pickle
import dlib
from model import create_model
from align import AlignDlib

model = '/home/legion/Documents/DeepLearning/face-recognition/Caffe model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt = '/home/legion/Documents/DeepLearning/face-recognition/Caffe model/deploy.prototxt.txt'
net = cv2.dnn.readNetFromCaffe(prototxt, model)
alignment = AlignDlib('models/landmarks.dat')
knn = pickle.load(open('svc_model.sav', 'rb'))
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

img = cv2.imread('Recog/dab.png',1)
# bb = alignment.getAllFaceBoundingBoxes(img)

def get_bb(img):
    list_bb = []
    (h, w) = img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(img, (300, 300)), 1.0,
    	(300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()
    for i in range(0, detections.shape[2]):
    	confidence = detections[0, 0, i, 2]
    	if confidence < 0.3:
    		continue
    	box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    	(startX, startY, endX, endY) = box.astype("int")
        bb = dlib.rectangle(left = startX, right = endX , top = startY, bottom = endY)
        list_bb.append(bb)
    return list_bb


bb = get_bb(img)
print(bb)
font = cv2.FONT_HERSHEY_SIMPLEX


for a in bb:
    t = a.top()
    l = a.left()
    r = a.right()
    b = a.bottom()
    # _img = img
    _img = img[t:b,l:r]
    # print(_img)
    cv2.imshow('img',_img)
    _img = alignment.align(96,_img,a,landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    # cv2.imshow('img',_img)
    _img = (_img / 255.).astype(np.float32)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
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
