import os
import cv2 as cv
import numpy as np

from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork

path = 'datasets/blah'

# Initialize computational graphs of both sub-networks
net_pose = HumanPoseIRNetwork()

net_gait = GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 2)

# Load pre-trained models
net_pose.restore('models/MPII+LSP.ckpt')
net_gait.restore('models/M+L-GRU-2.ckpt')


def get_frames(folder_path):
	pics = os.listdir(folder_path)
	pics.sort()
	frames = []
	for pic in pics:
		img = cv.imread(os.path.join(path,pic), cv.IMREAD_COLOR)
		img = cv.resize(img, (299,299))
		frames.append(img)
	frames = np.asarray(frames)
	return frames

def get_id_vectors(folder_path):
	frames = get_frames(folder_path)
	spatial_features = net_pose.feed_forward_features(frames)
	identification_vector = net_gait.feed_forward(spatial_features)
	id_vector = np.array([identification_vector[0], identification_vector[1][0][0], identification_vector[1][1][0]])
	del frames, identification_vector, spatial_features
	return id_vector


aravind_id_vectors = get_id_vectors(path)
# chakri_id_vectors  = get_id_vectors(os.path.join(path, 'chakri_predict.mp4'))
# print(aravind_id_vectors.shape)
print(aravind_id_vectors.shape)

np.save('blah_id_vector.npy', aravind_id_vectors )
# np.save('chakri_id_vector_test.npy',  chakri_id_vectors)