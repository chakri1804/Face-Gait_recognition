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

def recognise(img):
    bb = alignment.getLargestFaceBoundingBox(img)
    # img_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.INNER_EYES_AND_BOTTOM_LIP)
    if bb is not None:
        img_aligned = img[bb.left():bb.right(), bb.top():bb.bottom(), :]
        img_aligned = cv2.resize(img_aligned, (96,96))
        return bb , img_aligned
    else:
        return  None, None

def encoding(img):
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))
    return embedded

def predict(vec):
    return knn.predict(vec)


# img_path = 'Recog/image.jpeg'
cap = cv2.VideoCapture(0)
# cap.set(cv2.CV_CAP_PROP_FPS, 60)
# print(val)
# print(type(val))


while cap.isOpened():
    ret, img = cap.read()
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    bb , img_aligned = recognise(img)
    if bb is not None:
        encoded = encoding(img_aligned)
        val = predict(encoded)
        cv2.rectangle(img,(bb.left(),bb.top()),(bb.left()+bb.width(),bb.top()+bb.height()),(0,255,0),3)
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img,val[0],(bb.left(),bb.top()), font, 1,(255,255,255))
    cv2.imshow('img',img)
    k = cv2.waitKey(1) & 0xff
    if k == 'q': break

cap.release()
cv2.destroyAllWindows()
