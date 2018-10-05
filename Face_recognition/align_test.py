from align import AlignDlib
# import  numpy as np
import time
import cv2
# from PIL import Image

alignment = AlignDlib('models/landmarks.dat')

while True:
    img = cv2.imread('Recog/shanky.jpeg',1)
    start = time.time()
    bb = alignment.getLargestFaceBoundingBox(img)
    end = time.time()
    img_aligned = alignment.align(96, img, bb, landmarkIndices=AlignDlib.OUTER_EYES_AND_NOSE)
    print(end - start)
