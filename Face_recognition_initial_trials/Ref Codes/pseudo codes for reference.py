from keras.layers import Activation, Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

if __name__ == '__main__':    
    # input image dimensions
    img_rows, img_cols = 225, 225
    input_shape = (img_rows, img_cols, 1)

    model = Sequential()
    model.add(Convolution2D(64, 15, 15, input_shape=input_shape, subsample=(3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2)))

    model.compile(loss='hinge', optimizer='adadelta', metrics=['accuracy'])

    data = ImageDataGenerator()
    train_data = data.flow_from_directory(directory='dataset', color_mode='grayscale', target_size=(img_rows, img_cols))
    model.fit_generator(train_data, 100, 12)

    flow_from_directory(directory
 target_size=(256, 256), color_mode='rgb', classes=None, class_mode='categorical', batch_size=32, shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='png', follow_links=False, subset=None, interpolation='nearest')

    fit_generator(self, generator, steps_per_epoch=None, epochs=1, verbose=1, callbacks=None, validation_data=None, validation_steps=None, class_weight=None, max_queue_size=10, workers=1, use_multiprocessing=False, shuffle=True, initial_epoch=0)

    predict_generator(self, 
                      generator, 
                      steps=None, 
                      max_queue_size=10, 
                      workers=1, 
                      use_multiprocessing=False,
                      verbose=0)
