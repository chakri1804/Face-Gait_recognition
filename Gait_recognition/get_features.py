import numpy as np
import os
import pandas
from scipy.misc import imread, imresize

from human_pose_nn import HumanPoseIRNetwork
from gait_nn import GaitNetwork

path = 'images/Running_frontal'
pics = os.listdir(path)
pics.sort()
imgs = []

for pic in pics:
	img = imread(os.path.join(path, pic))
	img = imresize(img, [299,299])
	imgs.append(img)

video_frames = np.asarray(imgs, dtype=np.uint8)
print(video_frames.shape)
# Initialize computational graphs of both sub-networks
net_pose = HumanPoseIRNetwork()

net_gait = GaitNetwork(recurrent_unit = 'GRU', rnn_layers = 3)

# Load pre-trained models
net_pose.restore('models/MPII+LSP.ckpt')
net_gait.restore('models/M+L-GRU-2.ckpt')

# Create features from input frames in shape (TIME, HEIGHT, WIDTH, CHANNELS) 
spatial_features = net_pose.feed_forward_features(video_frames)
# np.savetxt('spatial_features.txt', spatial_features)
print(spatial_features.shape)
# print(spatial_features.shape)
# spacial_df = pandas.DataFrame(data=spatial_features)
# spacial_df.to_csv(path_or_buf='images/spatial_features.csv', sep=",")


# Process spatial features and generate identification vector 
identification_vector = net_gait.feed_forward(spatial_features)
print(identification_vector[0].shape)

print(type(identification_vector))
print(type(identification_vector[0]))
print(type(identification_vector[1]))

print("***************************************")
for blah in identification_vector[1]:
	print(blah[0].shape)
	print(type(blah))

# print("***************************************")
blah_tensor = np.array([identification_vector[0], identification_vector[1][0][0], identification_vector[1][1][0]])

print(blah_tensor.shape)
# identification_vector = np.asarray(identification_vector)
# print(identification_vector.shape)

# # np.savetxt('identification_vectors.txt', identification_vector)

# # print(identification_vector)
# id_df = pandas.DataFrame(data=identification_vector)
# id_df.to_csv(path_or_buf='images/identification_vectors.csv', sep=",", index=None, header=None)