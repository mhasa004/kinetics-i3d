import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import i3d
import random
import pickle
import argparse
import cv2

import warnings
warnings.filterwarnings("ignore")


class KineticsFeature(object):

	def resize_input(self, image):
		# retain aspect ratio. keep short side at 256
		min_side_len = 256
		image_w = image.shape[1]
		image_h = image.shape[0]
		if image_w <= min_side_len or image_h <= min_side_len:
			return image
		if image_h < image_w:
			image_h_new = min_side_len
			image_w_new = int(min_side_len * (image_w/image_h))
		else:
			image_w_new = min_side_len
			image_h_new = int(min_side_len * (image_h/image_w))

		new_image = cv2.resize(image, (image_h_new, image_w_new), cv2.INTER_LINEAR)
		return new_image


	def center_crop(self, image):
		x = int(image.shape[0] / 2) - int(self.imgsize / 2)
		y = int(image.shape[1] / 2) - int(self.imgsize / 2)
		return image[x:x+self.imgsize, y:y+self.imgsize, :]


	def process_frame_v2(self, frame, num_channel):
		frame = self.resize_input(frame)
		frame = self.center_crop(frame)
		frame = 2 * (frame / 255) - 1
		return frame[:, :, :num_channel]


	def load_i3d_model(self):

		if self.args.type in ['rgb', 'joint']:
			self.model_input_rgb = tf.placeholder(tf.float32, shape=(None, None, self.imgsize, self.imgsize, 3))
			with tf.variable_scope('RGB'):
				rgb_model = i3d.InceptionI3d(self.num_classes, spatial_squeeze=True, final_endpoint='Logits')
				logits_rgb, feature_rgb, _ = rgb_model(self.model_input_rgb, is_training=False, dropout_keep_prob=1.0)

			rgb_variable_map = {}
			for variable in tf.global_variables():
				if variable.name.split('/')[0] == 'RGB':
					rgb_variable_map[variable.name.replace(':0', '')] = variable
			rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
			rgb_saver.restore(self.sess, 'data/checkpoints/rgb_imagenet/model.ckpt')
			print('RGB checkpoint restored')

		if self.args.type in ['flow', 'joint']:
			self.model_input_flow = tf.placeholder(tf.float32, shape=(None, None, self.imgsize, self.imgsize, 2))
			with tf.variable_scope('Flow'):
				flow_model = i3d.InceptionI3d(self.num_classes, spatial_squeeze=True, final_endpoint='Logits')
				logits_flow, feature_flow, _ = flow_model(self.model_input_flow, is_training=False, dropout_keep_prob=1.0)

			flow_variable_map = {}
			for variable in tf.global_variables():
				if variable.name.split('/')[0] == 'Flow':
					flow_variable_map[variable.name.replace(':0', '')] = variable
			flow_saver = tf.train.Saver(var_list=flow_variable_map, reshape=True)
			flow_saver.restore(self.sess, 'data/checkpoints/flow_imagenet/model.ckpt')
			print('Flow checkpoint restored')

		if self.args.type == 'rgb':
			self.model_logits = logits_rgb
			self.model_feature = feature_rgb
		elif self.args.type == 'flow':
			self.model_logits = logits_flow
			self.model_feature = feature_flow
		else:
			self.model_logits = logits_rgb + logits_flow
			self.model_feature = feature_rgb + feature_flow

		self.model_prediction = tf.nn.softmax(self.model_logits)


	def extract_feature_joint(self, frames, flows):
		num_frames = frames.shape[0]
		if self.args.tw == -1:
			video_segs = np.expand_dims(frames, axis=0)
			flow_segs = np.expand_dims(flows, axis=0)
		else:
			num_segs = (num_frames - self.args.tw) // self.args.stride + 1
			video_segs = np.zeros(shape=(num_segs, self.args.tw, self.imgsize, self.imgsize, 3))
			flow_segs = np.zeros(shape=(num_segs, self.args.tw, self.imgsize, self.imgsize, 2))
			idx = 0
			for i in range(0, num_frames-self.args.tw, self.args.stride):
				video_segs[idx] = frames[i:i+self.args.tw, :, :, :]
				flow_segs[idx] = flows[i:i+self.args.tw, :, :, :]
				idx += 1
		
		feed_dict = {self.model_input_rgb: video_segs, self.model_input_flow: flow_segs}
		model_out = self.sess.run([self.model_feature, self.model_logits, self.model_prediction], feed_dict)

		return model_out


	def extract_feature_rgb(self, frames):
		num_frames = frames.shape[0]
		if self.args.tw == -1:
			video_segs = np.expand_dims(frames, axis=0)
		else:
			num_segs = (num_frames - self.args.tw) // self.args.stride + 1
			video_segs = np.zeros(shape=(num_segs, self.args.tw, self.imgsize, self.imgsize, 3))
			idx = 0
			for i in range(0, num_frames-self.args.tw, self.args.stride):
				video_segs[idx] = frames[i:i+self.args.tw, :, :, :]
				idx += 1
			print(video_segs.shape)
		
		feed_dict = {self.model_input_rgb: video_segs}
		model_out = self.sess.run([self.model_feature, self.model_logits, self.model_prediction], feed_dict)

		return model_out


	def extract_feature_flow(self, flows):
		num_frames = flows.shape[0]
		if self.args.tw == -1:
			flow_segs = np.expand_dims(flows, axis=0)
		else:
			num_segs = (num_frames - self.args.tw) // self.args.stride + 1
			flow_segs = np.zeros(shape=(num_segs, self.args.tw, self.imgsize, self.imgsize, 2))
			idx = 0
			for i in range(0, num_frames-self.args.tw, self.args.stride):
				flow_segs[idx] = flows[i:i+self.args.tw, :, :, :]
				idx += 1
		
		feed_dict = {self.model_input_flow: flow_segs}
		model_out = self.sess.run([self.model_feature, self.model_logits, self.model_prediction], feed_dict)

		return model_out


	def create_frame_generator(self, video_path, num_channel):
		num_unique_frames_batch = self.args.tw + (self.args.batch - 1) * self.args.stride
		frame_buffer = np.zeros(shape=(num_unique_frames_batch, self.imgsize, self.imgsize, num_channel))
		cap = cv2.VideoCapture(video_path)
		i = 0
		while cap.isOpened():
			ret, frame = cap.read()
			if not ret:
				if i >= self.args.tw:
					yield frame_buffer[:i]
				return
			frame = self.process_frame_v2(frame, num_channel=num_channel)
			frame_buffer[i] = frame
			if i == num_unique_frames_batch - 1:
				yield frame_buffer
				i = self.args.tw - self.args.stride
				frame_buffer[:i] = frame_buffer[-i:]
			else:
				i += 1


	def __call__(self):
		max_number_features = int(3 * 3600 * 30 / self.args.stride)   # assuming 3 hrs of video
		features = np.zeros(shape=(max_number_features, 1024))
		i = 0
		batch_no = 0
		if self.args.type == 'joint':
			for fr, fl in zip(
					self.create_frame_generator(self.args.video, num_channel=3),
					self.create_frame_generator(self.args.flow, num_channel=2)):
				feature, _, _ = self.extract_feature_joint(fr, fl)
				feature = np.squeeze(feature)
				if len(feature.shape) == 2:
					features[i:i+feature.shape[0]] = feature
					i += feature.shape[0]
				else:
					features[i] = feature
					i += 1
				batch_no += 1
				print('Batch no: %d' % batch_no)
			features = features[:i]

		elif self.args.type == 'rgb':
			for frame_buffer in self.create_frame_generator(self.args.video, num_channel=3):
				feature, _, _ = self.extract_feature_rgb(frame_buffer)
				feature = np.squeeze(feature)
				if len(feature.shape) == 2:
					features[i:i + feature.shape[0]] = feature
					i += feature.shape[0]
				else:
					features[i] = feature
					i += 1
			features = features[:i]
		elif self.args.type == 'flow':
			for frame_buffer in self.create_frame_generator(self.args.flow, num_channel=2):
				feature, _, _ = self.extract_feature_flow(frame_buffer)
				feature = np.squeeze(feature)
				if len(feature.shape) == 2:
					features[i:i + feature.shape[0]] = feature
					i += feature.shape[0]
				else:
					features[i] = feature
					i += 1
			features = features[:i]

		outfile = self.args.outfile + '_tw%d_stride%d.npy' % (self.args.tw, self.args.stride)
		np.save(outfile, features)


	def __init__(self, args):
		self.args = args

		if os.path.isfile(self.args.video) and os.path.isfile(self.args.flow):
			self.args.type = 'joint'
		elif os.path.isfile(self.args.video):
			self.args.type = 'rgb'
		elif os.path.isfile(self.args.flow):
			self.args.type = 'flow'
		else:
			raise ValueError

		self.imgsize = 224
		self.num_classes = 400	
		
		self.model_input_rgb = None
		self.model_input_flow = None
		self.model_logits = None
		self.model_feature = None
		self.model_prediction = None
		self.sess = tf.Session()
		self.load_i3d_model()


def main(args):
	kinetics_feature = KineticsFeature(args)
	kinetics_feature()


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Kinetics feature extraction')
	parser.add_argument('--video', action='store', default='')
	parser.add_argument('--flow', action='store', default='')
	parser.add_argument('--batch', default=8, type=int)
	parser.add_argument('--tw', default=16, type=int)
	parser.add_argument('--stride', default=8, type=int)
	parser.add_argument('--outfile', action='store', default='feature')
	targs = parser.parse_args()
	main(targs)
