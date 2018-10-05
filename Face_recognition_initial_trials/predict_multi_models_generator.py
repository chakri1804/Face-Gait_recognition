import keras
from keras.models import load_model
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

model = load_model('not_so_activated_35.h5')
path1 = 'vggface2_test/val/'
test_data = ImageDataGenerator(rescale=1./255)

validation_generator = test_data.flow_from_directory(
    path1,
    target_size=(100,80),
	batch_size=300
    )

res =   model.evaluate_generator(
            validation_generator,
            max_queue_size=30,
            verbose=1
        )
print('corrected_my_net_30.h5')
print res
