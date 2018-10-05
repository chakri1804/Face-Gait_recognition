import keras
from keras.models import load_model
import cv2
import numpy as np

model = load_model('/home/legion/Documents/DeepLearning/VGGFace2 dataset/Trained Models/corrected_my_net_40.h5')
model.summary()
# img = cv2.imread('0159_01.jpg')
# img = img/255.0
# img = np.reshape(img, (1,100,80,3))
# print model.predict_classes(img)
