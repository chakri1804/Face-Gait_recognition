import numpy as np
import glob, os
import pandas as pd
import numpy as np
import cv2 as cv 
import keras
from keras.applications.vgg19 import VGG19
from keras.layers import Input
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model

path = 'vggface2_test/test/'
path1 = 'vggface2_test/val/'
erects = pd.read_csv('bb_landmark/loose_bb_train.csv')
blah = np.asarray(erects)
num_epoch = 10

###########################
#       Define model      #
###########################

model = VGG19(
	include_top=True, 
	weights=None, 
	input_tensor=None, 
	input_shape=(100, 80, 3), 
	pooling=None, classes=500
	)

model.compile(
	optimizer = optimizers.adam(lr = 1e-4), 
	loss = 'categorical_crossentropy', 
	metrics = ['accuracy']
	)

model.summary()

###########################

train_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    path,
    target_size=(100,80),
	batch_size=200
	)

validation_generator = test_data.flow_from_directory(
    path1,
    target_size=(100,80),
	batch_size=300
    )

history = model.fit_generator(
			train_generator, 
			steps_per_epoch=1000,
			epochs=20,
			shuffle=True,
			verbose=1
			)

model.save('corrected_VGG19_20.h5')

########################
# Post-processing shit #
########################
res =   model.evaluate_generator(
            validation_generator, 
            max_queue_size=30,
            verbose=1
        )

print res

np.save("VGG19_Acc_20.npy", history.history['acc'])
np.save("VGG19_Loss_20.npy", history.history['loss'])

model = load_model('corrected_VGG19_20')

model.fit_generator(
			train_generator, 
			steps_per_epoch=1000,
			epochs=5,
			shuffle=True,
			verbose=1
			)

res =   model.evaluate_generator(
            validation_generator, 
            max_queue_size=30,
            verbose=1
        )
print res

model.save('corrected_VGG19_25.h5')

model = load_model('corrected_VGG19_25')

model.fit_generator(
			train_generator, 
			steps_per_epoch=1000,
			epochs=5,
			shuffle=True,
			verbose=1
			)

res =   model.evaluate_generator(
            validation_generator, 
            max_queue_size=30,
            verbose=1
        )
print res

model.save('corrected_VGG19_30.h5')