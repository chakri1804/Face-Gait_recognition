import os
import cv2 as cv
import numpy as np
import pandas
from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork

path = 'datasets/WalkingVideos'

videos  = os.listdir(path)
videos.sort()

# Initialize computational graphs of both sub-networks
net_pose = HumanPoseIRNetwork()

# net_gait = GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 2)

# Load pre-trained models
net_pose.restore('models/MPII+LSP.ckpt')
# net_gait.restore('models/H3.6m-GRU-1.ckpt')


def get_spatial_features(vid_path):
	_frames = []
	cap = cv.VideoCapture(vid_path)
	success, _frame = cap.read()
	# count = 0
	success = True
	while success:
		_frame = cv.resize(_frame, (299,299))
		_frames.append(_frame)
		success, _frame = cap.read()
		# count += 1
	cap.release()
	del cap
	_frames = np.asarray(_frames)
	_spatial_features = net_pose.feed_forward_features(_frames)
	# identification_vector = net_gait.feed_forward(_spatial_features)
	# _id_vector = np.array([identification_vector[0], identification_vector[1][0][0], identification_vector[1][1][0]])
	return _spatial_features
labels = []

for video in videos:
	spatial_features = get_spatial_features(os.path.join(path, video))
	np.save(video[:-4]+'npy', spatial_features)
	label = video[0:4]
	labels.append(label)
	print(video)

labels = pandas.DataFrame(labels)
labels.to_csv('labels.csv', mode='a', index=False, header=False)
