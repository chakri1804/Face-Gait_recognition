from __future__ import division
import os
import keras
from keras.layers import BatchNormalization
from keras.layers import Dense, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D,AveragePooling2D,GaussianNoise
from keras.models import Sequential
from keras import regularizers
import numpy as np
import cv2
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report



def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        if any([filename.endswith(x) for x in ['.jpeg', '.jpg','.JPG','.tif']]):
            img = cv2.imread(os.path.join(folder, filename))
            img = cv2.medianBlur(img,3)
        if img is not None:
            images.append(img)
    return images



folders = [
    '1',
    '2',
    '3'
]
num_classes=len(folders)
df={}
for folder in folders:
    df[folder] = np.array(load_images_from_folder(folder))
    print(df[folder].shape)
for folder in folders:
    df[folder] = df[folder].reshape(len(df[folder]),256*256*3)
X=[]
y=[]
k=0
for i in folders:
    for elem in df[i]:
        X.append(elem)
        y.append(k)
    k=k+1
X=np.array(X)
y=np.array(y)
X=X/255*1.0

print(np.max(X))
print(X.shape)
print(y.shape)



num=X.shape[0]
X_reshaped=X.reshape(num,256,256,3)
print(X_reshaped.shape)

print(y.shape)



X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.15,random_state=143)

batch_size=16
epochs=20

y_train=keras.utils.to_categorical(y_train,num_classes)
y_test1=keras.utils.to_categorical(y_test,num_classes)



Model1 = Sequential()
random.seed(19)

Model1.add(Conv2D(32, (1, 3), strides=(1,1), padding='same', activation='relu', input_shape=(256,256,3)))
Model1.add(Conv2D(32, (3, 1), strides=(1,1), padding='same', activation='relu'))
Model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())


Model1.add(Conv2D(64, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(64, (3, 1), padding='same', activation='relu'))
Model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())


Model1.add(Conv2D(128, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(128, (3, 1), padding='same', activation='relu'))
Model1.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())


Model1.add(Conv2D(128, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(128, (3, 1), padding='same', activation='relu'))
Model1.add(BatchNormalization())

Model1.add(Conv2D(256, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(256,(3,1),padding='same',activation='relu'))
Model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())

Model1.add(Conv2D(512, (1, 3), padding='same', activation='relu'))
Model1.add(Conv2D(512,(3,1),padding='same',activation='relu'))
Model1.add(BatchNormalization())

Model1.add(Conv2D(512, (1,3), padding='same', activation='relu'))
Model1.add(Conv2D(512,(3,1),padding='same',activation='relu'))
Model1.add(BatchNormalization())


Model1.add(GaussianNoise(stddev=0.001))
Model1.add(AveragePooling2D(pool_size=(3, 3), strides=(2, 2)))
Model1.add(BatchNormalization())

Model1.add(Flatten())
Model1.add(Dense(1024, activation='relu'))
Model1.add(Dropout(0.5))
Model1.add(Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
Model1.add(Dropout(0.3))

Model1.add(Dense(num_classes, activation='softmax'))
Model1.summary()



Model1.compile(loss=keras.losses.categorical_crossentropy,optimizer=keras.optimizers.Adam(lr=1e-5),metrics=['accuracy'])
History=Model1.fit(X_train,y_train,batch_size=batch_size,validation_data=(X_test, y_test1),epochs=20,verbose=1)


score=Model1.evaluate(X_test,y_test1)
print(score[0])
print(score[1])
predict=Model1.predict_classes(X_test)
print(confusion_matrix(y_test,predict))
Model1.save("3_classs.h5")

target_names=['normal','DR','ARMD']
print(classification_report(y_test,predict,target_names=target_names))

# np.save("Acc_custom.npy", History.history['acc'])
# np.save("Val_Acc_custom.npy", History.history['val_acc'])
# np.save("Loss_custom.npy", History.history['loss'])
# np.save("Val_Loss_custom.npy", History.history['val_loss'])
