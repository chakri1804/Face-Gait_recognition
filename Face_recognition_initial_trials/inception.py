import os
import keras
from keras.models import *
from keras.layers import *
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras import optimizers

############################
# Useful funcs
############################
# def save_plots(history):
# 	np.save("Acc_incept_basic.npy", history.history['acc'])
# 	np.save("val.Acc_incept_basic.npy", history.history['val_acc'])
# 	np.save("Loss_incept_basic.npy", history.history['loss'])
# 	np.save("val.Loss_incept_basic.npy", history.history['val_loss'])

path = 'vggface2_test/test/'
path1 = 'vggface2_test/val/'

############################
# Net Blocks
############################

###########################
# inception_block
###########################

def inception_block(x,filters_1,filters_2,filters_3,filters_4):
	inp = x
	x_1=Conv2D(filters_1,(1,1))(inp)
	x_1=BatchNormalization()(x_1)
	x_1=LeakyReLU(alpha=0.01)(x_1)
	x_1=ZeroPadding2D(padding=(1,1))(x_1)
	x_1=Conv2D(filters_1,(3,3))(x_1)
	x_1=BatchNormalization()(x_1)
	x_1=LeakyReLU(alpha=0.01)(x_1)
	# x_1=Conv2D(filters_1,(1,3),padding='same')(x_1)

	x_2=Conv2D(filters_1,(1,1))(inp)
	x_2=BatchNormalization()(x_2)
	x_2=LeakyReLU(alpha=0.01)(x_2)
	x_2=ZeroPadding2D(padding=(1,1))(x_2)
	x_2=Conv2D(filters_1,(5,5))(x_2)
	x_2=BatchNormalization()(x_2)
	x_2=LeakyReLU(alpha=0.01)(x_2)
	x_2=ZeroPadding2D(padding=(1,1))(x_2)

	# x_3=MaxPooling2D(pool_size=(2,2), strides=(2,2))(inp)
	x_3=Conv2D(filters_3,(1,1),padding='same')(inp)
	x_3=BatchNormalization()(x_3)
	x_3=LeakyReLU(alpha=0.01)(x_3)
	# x_3=ZeroPadding2D(padding=(1,1))

	x_4=Conv2D(filters_4,(1,1),padding='same')(inp)
	x_4=BatchNormalization()(x_4)
	x_4=LeakyReLU(alpha=0.01)(x_4)
	# x_4=ZeroPadding2D(padding=(1,1))

	concat=concatenate([x_1,x_2,x_3,x_4])
	# bottleneck=Conv2D(np.int64(total/2),(1,1),padding='same')(concat)

	return concat

##############################
#Conv->MaxPool->Batch-norm
##############################

def gen_block_1(x,filter_size,pool_size,number_of_filters,strides,padding):
	x=Conv2D(number_of_filters,(filter_size,1),padding='same')(x)
	x=Conv2D(number_of_filters,(1,filter_size),padding='same')(x)
	x=Conv2D(number_of_filters,(filter_size,filter_size),padding='same')(x)
	x=LeakyReLU(alpha=0.01)(x)
	if padding==1:
		x=MaxPooling2D(pool_size=pool_size,padding='same',strides=strides)(x)
	else:
		x=MaxPooling2D(pool_size=pool_size,strides=strides)(x)
	x=BatchNormalization()(x)
	return x

# ##############################
# #Max pool->Conv2D->Batch-norm
# #To use after Inception Blocks
# ##############################
#
# def gen_block_2(x,filter_size,pool_size,number_of_filters,strides,padding):
# 	if padding==1:
# 		x=MaxPooling2D(pool_size=pool_size,padding='same',strides=strides)(x)
# 	else:
# 		x=MaxPooling2D(pool_size=pool_size,strides=strides)(x)
# 	x=Conv2D(number_of_filters,(filter_size,1),padding='same')(x)
# 	x=Conv2D(number_of_filters,(1,filter_size),padding='same')(x)
# 	x=Conv2D(number_of_filters,(filter_size,filter_size),padding='same')(x)
# 	x=LeakyReLU(alpha=0.01)(x)
# 	x=BatchNormalization()(x)
# 	return x

################################
# DENSE layers
################################
def Dense_layer(x,fil_1,fil_2,classes):
	x=Dense(fil_1,kernel_regularizer=regularizers.l1(0.001))(x)
	x=LeakyReLU(alpha=0.001)(x)
	x=Dropout(0.2)(x)
	x=Dense(fil_2,kernel_regularizer=regularizers.l1(0.001))(x)
	x=LeakyReLU(alpha=0.001)(x)
	x=Dropout(0.2)(x)
	x=Dense(classes, activation='softmax')(x)
	return x

inputs = Input(shape=(100,80,3))
###############################
###############################
# x = inception_block(inputs,16,16,16,16)
# # x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
# x = gen_block_2(x,3,(3,3),32,(2,2), padding=1)
# x = gen_block_1(x,3,(3,3),64,(2,2), padding=1)
# x = inception_block(x, 128,128,128,128)
# x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
# x = gen_block_1(x,3,(3,3),256,(2,2), padding=1)
# x = gen_block_1(x,3,(3,3),512,(2,2), padding=1)
# x = Flatten()(x)
# x = Dense_layer(x,2048,1024,500)
#
# model=Model(inputs=inputs,outputs=x)
# model.summary()


x = gen_block_1(inputs,3,(3,3),8,(2,2),padding=0)
x = inception_block(x,16,16,16,16)
x = inception_block(x,32,32,32,32)
x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
x = inception_block(x,64,64,64,64)
x = inception_block(x,64,64,64,64)
x = MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
x = inception_block(x,128,128,128,128)
x = inception_block(x,128,128,128,128)
x = AveragePooling2D(pool_size=(3,3),strides=(2,2))(x)
x = inception_block(x,256,256,256,256)
x = AveragePooling2D(pool_size=(3,3),strides=(2,2))(x)
x = Flatten()(x)
x = Dense_layer(x,512,512,500)

model = Model(inputs=inputs,outputs=x)
model.summary()

model.compile(
	optimizer = optimizers.adam(lr = 1e-5),
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
	)


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
	batch_size=270
    )

history = model.fit_generator(
			train_generator,
			steps_per_epoch=1000,
			epochs=30,
			shuffle=True,
			verbose=1,
			validation_data=validation_generator,
			validation_steps=111
			)

model.save('inception.h5')

res =   model.evaluate_generator(
            validation_generator,
            max_queue_size=35,
            verbose=1
        )

print(res)

model = load_model('inception_30.h5')


model.compile(
	optimizer = optimizers.adam(lr = 1e-4),
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
	)
history = model.fit_generator(
			train_generator,
			steps_per_epoch=1000,
			epochs=10,
			shuffle=True,
			verbose=1,
			validation_data=validation_generator,
			validation_steps=111,
			initial_epoch=36
			)

model.save('inception_45.h5')

res =   model.evaluate_generator(
            validation_generator,
            max_queue_size=30,
            verbose=1
        )

print(res)
