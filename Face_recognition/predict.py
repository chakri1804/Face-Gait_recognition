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
