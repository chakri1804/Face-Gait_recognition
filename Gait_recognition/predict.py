import pickle
model = pickle.load(open('gait.pickle', 'rb'))

import numpy as np
test = np.load('chakri_id_vector_test.npy')
test = np.reshape(test, (1,1536))

print(model.predict(test))