from __future__ import division
import os
import keras
from keras.utils import plot_model
from keras.models import *
from keras.layers import *
from keras import regularizers
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from PIL import Image
from scipy.misc import imread,imresize

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.JPG','.tif']]):
            img = imread(os.path.join(folder, filename))
            if img is not None:
                images.append(img)
    return np.array(images)



folders = [
	'normal_aug',
    'dr_aug',
    'amd_aug'
]
num_classes=len(folders)
df={}
for folder in folders:
    df[folder] = load_images_from_folder(folder)
for folder in folders:
    df[folder] = df[folder].reshape(len(df[folder]),256*256*3)
X=[]
y=[]
for i in range(0,len(folders)):
    for elem in df[folders[i]]:
        X.append(elem)
        y.append(int(i))
X=np.array(X)
y=np.array(y)
print(2+2)
X=X/255*1.0

print(np.max(X))
print(X.shape)
print(y.shape)



num=X.shape[0]
X_reshaped=X.reshape(num,256,256,3)
print(X_reshaped.shape)

print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2,random_state=143)

batch_size=16
epochs=20

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test1=keras.utils.to_categorical(y_test,num_classes)



# Below part is the model
# below function generates an inception block 
def inception_block(x,num_filters_1,num_filters_2,num_filters_3,num_filters_4):
	total=num_filters_1+num_filters_2+num_filters_3+num_filters_4
	inp=x
	x_1=Conv2D(num_filters_1,(1,1),padding='same')(inp)
	x_1=Conv2D(num_filters_1,(3,1),padding='same')(x_1)
	x_1=Conv2D(num_filters_1,(1,3),padding='same')(x_1)
	x_1=LeakyReLU(alpha=0.01)(x_1)
	x_1=BatchNormalization()(x_1)

	x_2=Conv2D(num_filters_2,(1,1),padding='same')(inp)
	x_2=Conv2D(num_filters_2,(5,1),padding='same')(x_2)
	x_2=Conv2D(num_filters_2,(1,5),padding='same')(x_2)
	x_2=LeakyReLU(alpha=0.01)(x_2)
	x_2=BatchNormalization()(x_2)

	x_3=MaxPooling2D(pool_size=(3,3),padding="same",strides=(1,1))(inp)
	x_3=Conv2D(num_filters_3,(1,1),padding='same',strides=(1,1))(x_3)
	x_3=LeakyReLU(alpha=0.01)(x_3)
	x_3=BatchNormalization()(x_3)

	x_4=Conv2D(num_filters_4,(1,1),padding='same')(inp)
	x_4=BatchNormalization()(x_4)

	concat=concatenate([x_1,x_2,x_3,x_4])
	bottleneck=Conv2D(np.int64(total/2),(1,1),padding='same')(concat)

	return bottleneck

# below piece of code generates conv2d->leaky_Relu->Max pool->Batch_norm
def general_layer(x,filter_size,pool_size,number_of_filters,strides,padding):
    x=Conv2D(number_of_filters,(filter_size,1),padding='same')(x)
    x=Conv2D(number_of_filters,(1,filter_size),padding='same')(x)
    x=LeakyReLU(alpha=0.01)(x)
    if padding==1:
    	x=MaxPooling2D(pool_size=pool_size,padding='same',strides=strides)(x)
    else:
    	x=MaxPooling2D(pool_size=pool_size,strides=strides)(x)
    x=BatchNormalization()(x)
    return x

#Max pool->Conv2D->Batch-norm
def general_layer_1(x,filter_size,pool_size,number_of_filters,strides,padding):
    if padding==1:
    	x=MaxPooling2D(pool_size=pool_size,padding='same',strides=strides)(x)
    else:
    	x=MaxPooling2D(pool_size=pool_size,strides=strides)(x)
    x=Conv2D(number_of_filters,(filter_size,1),padding='same')(x)
    x=Conv2D(number_of_filters,(1,filter_size),padding='same')(x)
    x=LeakyReLU(alpha=0.01)(x)
    x=BatchNormalization()(x)
    return x

#Dense layer with dropouts
def Dense_layer(x,num_classes):
	x=Dense(512,activation="relu",kernel_regularizer=regularizers.l1(0.001))(x)
	x=Dropout(0.5)(x)
	x=Dense(512,activation="relu",kernel_regularizer=regularizers.l1(0.001))(x)
	x=Dropout(0.5)(x)
	x=Dense(num_classes,activation="softmax")(x)
	return x

inputs=Input(shape=(256,256,3))

#x=general_layer(inputs,3,pool_size=(3,3),number_of_filters=32,strides=(2,2),padding=0)
#x=general_layer(x,3,pool_size=(3,3),number_of_filters=32,strides=(2,2),padding=0)

#x=general_layer(inputs,3,pool_size=(3,3),number_of_filters=64,strides=(2,2),padding=0)
#x=general_layer(x,3,pool_size=(3,3),number_of_filters=64,strides=(2,2),padding=0)

x=inception_block(inputs,16,16,16,16)
x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
x=inception_block(x,32,32,32,32)
x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
x=inception_block(x,64,64,64,64)
x=general_layer_1(x,3,pool_size=(3,3),number_of_filters=128,strides=(2,2),padding=0)
x=general_layer(x,3,pool_size=(3,3),number_of_filters=256,strides=(2,2),padding=0)
x=MaxPooling2D(pool_size=(3,3),strides=(1,1))(x)
x=inception_block(x,256,256,256,256)
#x=general_layer_1(x,3,pool_size=(3,3),number_of_filters=512,strides=(2,2),padding=0)
#x=general_layer(x,3,pool_size=(3,3),number_of_filters=512,strides=(2,2),padding=0)
x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
x=GaussianNoise(stddev=0.001)(x)
x=GlobalAveragePooling2D()(x)
x=Dense(512,activation="relu",kernel_regularizer=regularizers.l1(0.001))(x)
x=Dense(3,activation="softmax")(x)

model=Model(inputs=inputs,outputs=x)
# summarize layers
print(model.summary())
# plot graph
model.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=1e-5),metrics=['accuracy'])
History=model.fit(X_train,y_train,batch_size=batch_size,validation_data=(X_test, y_test1),epochs=20,verbose=1)


score=model.evaluate(X_test,y_test1)
print(score[0])
print(score[1])

np.save("Acc_3.npy", History.history['acc'])
np.save("Val_Acc_3.npy", History.history['val_acc'])
np.save("Loss_3.npy", History.history['loss'])
np.save("Val_Loss_3.npy", History.history['val_loss'])

predict=model.predict(X_test)
predict=np.argmax(predict,axis=1)
print(confusion_matrix(y_test,predict))
# Model1.save("custom_cnn.h5")
model.save("norm,dr,amd.h5")
target_names=['normal','dr','amd']
print(classification_report(y_test,predict,target_names=target_names))
#plot_model(model,to_file='model.png')
#model.summary()







