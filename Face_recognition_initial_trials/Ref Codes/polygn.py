import cv2
import numpy as np

## "n000002/0001_01",
# 239.4643,276.5054,
# 349.8496,282.2599,
# 311.2406,354.8974,
# 236.0358,391.3312,
# 342.1541,392.2661

# "n000002/0001_01",161,140,224,324
img = cv2.imread('0001_01.jpg')
pts = np.array([[239.4643,276.5054],[349.8496,282.2599],[311.2406,354.8974],[236.0358,391.3312],[342.1541,392.2661]], np.int32)
pts = pts.reshape((-1,1,2))
cv2.polylines(img,[pts],True,(0,255,255))
print(img.shape)
cv2.rectangle(img,(161,140),(160+224,140+324),(0,255,0),3)

cv2.imshow('ploy',img)
cv2.waitKey(0)
cv2.destroyAllWindows()