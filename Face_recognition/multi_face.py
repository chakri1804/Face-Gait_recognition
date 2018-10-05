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
    _img = cv2.imread(path)
    _faces = []
    _bb = alignment.getAllFaceBoundingBoxes(_img)
    print _bb
    print _img.shape
    for box in _bb:
        print box
        print box.left(), box.top()
        face = _img[box.left():(box.left()+box.width()), box.top():(box.top()+box.height()),:]
        print face.shape
        face = cv2.resize(face, (96,96))
        _faces.append(face)
        # img_aligned = alignment.align(96, img, box)
    return _bb , _faces

def encoding(img):
    img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))
    return embedded

def predict(vec):
    return knn.predict(vec)


img_path = 'Recog/lul.jpg'
bb , faces = recognise(img_path)
embeddings = [encoding(face) for face in faces]
vals = [predict(embedding) for embedding in embeddings]
img1 = cv2.imread('Recog/lul.jpg',1)
print img1.shape
print(vals)
print(type(vals))
for i in range(0, len(vals)):
    cv2.rectangle(img1,(bb[i].left(),bb[i].top()),(bb[i].left()+bb[i].width(),bb[i].top()+bb[i].height()),(0,255,0),3)
    cv2.putText(img1,vals[i],(bb[i].left(),bb[i].top()), font, 1,(255,255,255))
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.imshow('img', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()
