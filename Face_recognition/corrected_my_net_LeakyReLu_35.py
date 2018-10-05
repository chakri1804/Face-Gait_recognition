import numpy as np
# import glob, os
import pandas as pd
# import numpy as np
# import cv2 as cv
import keras
from keras.layers import BatchNormalization
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,GaussianNoise
from keras.layers import LeakyReLU
from keras.models import Sequential
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras import optimizers
# from keras.applications.vgg19 import VGG19
# from sklearn.model_selection import train_test_split


path = 'aligned'
path1 = 'val'
# erects = pd.read_csv('bb_landmark/loose_bb_train.csv')
# blah = np.asarray(erects)
# num_epoch = 10
###########################
#Define model
###########################

# model = InceptionV3(weights = None, include_top = True, input_shape = (100, 80, 1), classes = 2)

# model = VGG19(
# 	include_top=True,
# 	weights=None,
# 	input_tensor=None,
# 	input_shape=(100, 80, 1),
# 	pooling=None, classes=500
# 	)

model = Sequential()

model.add(Conv2D(3, (4,5), strides=(1,1), padding='same', input_shape=(100,80,3)))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(32, (1,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(32, (3,1), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(BatchNormalization())

model.add(Conv2D(64, (1,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, (3,1), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(128, (1,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, (3,1), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(256, (1,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(256, (3,1), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(256, (3,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Conv2D(512, (1,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(512, (3,1), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model.add(BatchNormalization())

model.add(Flatten())
model.add(Dense(2048))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.4))
model.add(Dense(1024, kernel_regularizer=regularizers.l2(0.01)))
model.add(LeakyReLU(alpha=0.1))
model.add(Dropout(0.4))

model.add(Dense(500, activation='softmax'))
# model.summary()

##########################
model.compile(
	optimizer = optimizers.adam(lr = 1e-4),
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
	)

model.summary()
###########################
# Train = np.load("label.npy")
# # print Train.shape
# Test = np.load("test_labels.npy")
# print Test.shape

# data = ImageDataGenerator()
# train_data = data.flow_from_directory(
# 	directory=path,
# 	color_mode='grayscale',
# 	target_size=(100,80),
# 	batch_size=400
# 	)


train_data = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

test_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    path,
    target_size=(100,80),
	batch_size=138
	)

validation_generator = test_data.flow_from_directory(
    path1,
    target_size=(100,80),
	batch_size=300
    )

history = model.fit_generator(
			train_generator,
			steps_per_epoch=1000,
			epochs=35,
			shuffle=True,
			verbose=1,
			validation_data=validation_generator,
			validation_steps=111
			)

model.save('corrected_my_net_LeakyReLu_35.h5')

# res =   model.evaluate_generator(
#             validation_generator,
#             max_queue_size=30,
#             verbose=1
#         )

# print res

np.save("mynet_Acc_LeReLu_35.npy", history.history['acc'])
np.save("mynet_Acc.val_LeReLu_35.npy", History.history['val_acc'])
np.save("mynet_Loss_LeReLu_35.npy", history.history['loss'])
np.save("mynet_Loss.val_LeReLu_35.npy", History.history['val_loss'])

# model.save('VGG19.h5')
