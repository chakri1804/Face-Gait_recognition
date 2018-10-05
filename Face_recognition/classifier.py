from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.svm import SVR
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import pandas
import os
import pickle

path = 'embedding/'
# x = np.array([[]])
# y = np.array([])
x = []
y = []

for csv_file in os.listdir(path):
	df = pandas.read_csv(os.path.join(path,csv_file), header=None)
	vectors = np.array(df)
	x.extend(vectors)
	y.extend(len(vectors)*[csv_file[0:-4]])

x = np.asarray(x)
y = np.asarray(y)

# encoder = LabelEncoder()
#
# encoder.fit(y)
# y = encoder.transform(y)
print(y)
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

knn = KNeighborsClassifier(n_neighbors=2, metric='euclidean')
svc = LinearSVC()
svr_rbf = SVR(kernel='rbf',C=1,gamma=0.1)

knn.fit(X_train, y_train)
svc.fit(X_train, y_train)
# svr_rbf.fit(X_train, y_train)

acc_knn = accuracy_score(y_test, knn.predict(X_test))
acc_svc = accuracy_score(y_test, svc.predict(X_test))
# acc_svr = accuracy_score(y_test, svr_rbf.predict(X_test))
print(acc_knn)
print(acc_svc)
# print(f'SVR accuracy = {acc_svr}')


pickle.dump(knn, open('knn_model.sav', 'wb'))
pickle.dump(svc, open('svc_model.sav', 'wb'))

# loaded_model = pickle.load(open(filename, 'rb'))
