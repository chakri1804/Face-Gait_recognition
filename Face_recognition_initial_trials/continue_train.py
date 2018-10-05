import keras
from keras.layers import *
from keras.models import *
from keras import regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

path = 'vggface2_test/test/'
path1 = 'vggface2_test/val/'
model = load_model('inception.h5')

model.compile(
	optimizer = optimizers.adam(lr = 1e-4),
	loss = 'categorical_crossentropy',
	metrics = ['accuracy']
	)

model.summary()

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
			epochs=25,
			shuffle=True,
			verbose=1,
			validation_data=validation_generator,
			validation_steps=111,
			)

model.save('inception_40.h5')

# res =   model.evaluate_generator(
#             validation_generator,
#             max_queue_size=30,
#             verbose=1
#         )

# print res

# np.save("30_Acc_incept_basic.npy", history.history['acc'])
# np.save("30_val.Acc_incept_basic.npy", history.history['val_acc'])
# np.save("30_Loss_incept_basic.npy", history.history['loss'])
# np.save("30_val.Loss_incept_basic.npy", history.history['val_loss'])
