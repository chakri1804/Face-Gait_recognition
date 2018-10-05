############################################################

#Inspired From VGG19 net but added some modifications to 
#reduce params and some random ass convolutions and dropouts 
#in between to reduce overfitting issues on long epoch runs

############################################################

import keras
from keras.layers import *
from keras.models import Sequential
from keras import regularizers
import numpy as np

num_classes = #ask#
shape = #ask#

model = Sequential()

model.add(Conv2D(3, (4,5), strides=(1,1), padding='same', activation='relu', input_shape=shape)
model.add(Conv2D(32, (1,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(32, (3,1), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(BatchNormalization())

model.add(Conv2D(64, (1,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3,1), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(BatchNormalization())

model.add(Conv2D(128, (1,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(128, (3,1), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(BatchNormalization())

model.add(Conv2D(256, (1,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(256, (3,1), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(BatchNormalization())

model.add(Conv2D(512, (1,3), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(512, (3,1), strides=(1,1), padding='same', activation='relu'))
model.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.2))

model.add(Dense(num_classes, activation='softmax'))
model.summary()

##########################
#has 3 billion params :/
##########################

model1 = Sequential()

model1.add(Conv2D(3, (4,5), strides=(1,1), padding='same', activation='relu', input_shape=(100,80,3)))
model1.add(Conv2D(32, (1,3), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(32, (3,1), strides=(1,1), padding='same', activation='relu'))
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model1.add(BatchNormalization())

model1.add(Conv2D(64, (1,3), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(64, (3,1), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(64, (3,3), strides=(1,1), padding='same', activation='relu'))
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model1.add(BatchNormalization())

model1.add(Conv2D(128, (1,3), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(128, (3,1), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(128, (3,3), strides=(1,1), padding='same', activation='relu'))
model1.add(MaxPooling2D(pool_size=(3, 3), strides=(1, 1)))
model1.add(BatchNormalization())

model1.add(Conv2D(256, (1,3), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(256, (3,1), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(256, (3,3), strides=(1,1), padding='same', activation='relu'))
model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model1.add(BatchNormalization())

model1.add(Conv2D(512, (1,3), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(512, (3,1), strides=(1,1), padding='same', activation='relu'))
model1.add(Conv2D(512, (3,3), strides=(1,1), padding='same', activation='relu'))
model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
model1.add(BatchNormalization())

model.add(GlobalAveragePooling2D())

model1.add(Flatten())
model1.add(Dense(2048, activation='relu'))
model1.add(Dropout(0.2))
model1.add(Dense(2048, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model1.add(Dropout(0.2))

model1.add(Dense(500, activation='softmax'))
model1.summary()

#######################
#Has 402 million params
#######################