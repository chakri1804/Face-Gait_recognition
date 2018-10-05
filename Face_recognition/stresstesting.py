from model import create_model
import numpy as np
import time

nn4_small2_pretrained = create_model()
nn4_small2_pretrained.load_weights('weights/nn4.small2.v1.h5')


def encoding(img):
    # img = (img / 255.).astype(np.float32)
    # obtain embedding vector for image
    embedded = nn4_small2_pretrained.predict(np.expand_dims(img, axis=0))
    return embedded

i = 0
while(i<10):
    start = time.time()
    rand_shit = np.random.randint(0,255,size=(96,96,3))/255.
    encoded = encoding(rand_shit)
    end = time.time()
    # print(start)
    # print(end)
    print(end - start)
    # print("Loop Ended")
    # print(" ")
    i = i+1
