import numpy as np
import cv2

video_file = '/mnt/research-6f/mhasan/data/vsd/movies/ForrestGump.mp4'
feature_file = '/mnt/research-6f/mhasan/data/vsd/i3d/ForrestGump.i3d_tw16_stride8.npy'

cap = cv2.VideoCapture(video_file)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
print('Frame count: %d' % frame_count)

features = np.load(feature_file)
print('Total features: %d' % features.shape[0])

tw = 16
stride = 8
num_features = (frame_count - tw) / stride + 1
print('Number of features to be: %d' % num_features)
