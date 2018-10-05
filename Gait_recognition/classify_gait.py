from sklearn import svm
import numpy as np
import pickle
model = svm.SVC(kernel='linear', C=1, gamma=1) 

aravind_id_vectors = np.load('chakri_id_vector.npy')
chakri_id_vectors = np.load('aravind_id_vector.npy')

X = [np.reshape(aravind_id_vectors,(1536)), np.reshape(chakri_id_vectors,(1536))]

y = ['aravind','chakri']

model.fit(X, y)
model.score(X, y)

pickle.dump(model, open('gait.pickle', 'wb'))