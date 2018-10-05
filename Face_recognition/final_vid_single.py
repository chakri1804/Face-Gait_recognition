import os
import numpy as np
import cv2
import dlib
from align import AlignDlib
from model import create_model
import pickle

alignment = AlignDlib('models/landmarks.dat')
model = '/home/legion/Documents/DeepLearning/face-recognition/Caffe model/res10_300x300_ssd_iter_140000.caffemodel'
prototxt = '/home/legion/Documents/DeepLearning/face-recognition/Caffe model/deploy.prototxt.txt'
nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')

knn = pickle.load(open('knn_model.sav', 'rb'))
svc = pickle.load(open('svc_model.sav', 'rb'))

# landmark indices
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

net = cv2.dnn.readNetFromCaffe(prototxt, model)


def recognise(img):
    # BB = []
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
        BB = bb
    img_aligned = alignment.align(96, img, BB, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if img_aligned is not None:
        return bb , img_aligned
    else:
        return None, None

def encoding(img):
    img = (img / 255.).astype(np.float32)
    embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))
    return embedded

def predict(vec):
    return knn.predict(vec)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, img = cap.read()
    _bb , img_aligned = recognise(img)
    if (_bb is not None) and (img_aligned is not None):
        encoded = encoding(img_aligned)
        val = predict(encoded)
        cv2.rectangle(img,(_bb.left(),_bb.top()),(_bb.left()+_bb.width(),_bb.top()+_bb.height()),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,val[0],(_bb.left(),_bb.top()), font, 1,(255,255,255))
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 'q': break

cap.release()
cv2.destroyAllWindows()
