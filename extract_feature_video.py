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

        new_image = cv2.resize(image, [image_h_new, image_w_new], cv2.INTER_LINEAR)
        return new_image


    def read_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_np = np.zeros(shape=(frame_count, self.imgsize, self.imgsize, 3), dtype=np.float32)
        i = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            video_np[i, :, :, :] = self.process_frame(frame)
            i += 1
        cap.release()

        return video_np


    def read_flows(self, flow_path):
        cap = cv2.VideoCapture(flow_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        flow_np = np.zeros(shape=(frame_count, self.imgsize, self.imgsize, 2), dtype=np.float32)
        i = 0
        while cap.isOpened():
            ret, flow = cap.read()
            if not ret:
                break
            flow_np[i, :, :, :] = self.process_flow(flow)
            i += 1
        cap.release()

        return flow_np


    def center_crop(self, image):
        x = int(image.shape[0] / 2) - int(self.imgsize / 2)
        y = int(image.shape[1] / 2) - int(self.imgsize / 2)
        return image[x:x+self.imgsize, y:y+self.imgsize, :]


    def process_frame(self, frame):
        frame = self.resize_input(frame)
        frame = self.center_crop(frame)
        frame = 2 * (frame / 255) - 1
        return frame


    def process_flow(self, flow):
        flow = self.resize_input(flow)
        flow = self.center_crop(flow)
        flow = 2 * (flow / 255) - 1
        return flow[:, :, :2]


    def load_i3d_model(self):

        if self.args.type in ['rgb', 'joint']:
            self.model_input_rgb = tf.placeholder(tf.float32, shape=(None, self.args.tw, self.imgsize, self.imgsize, 3))
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
            self.model_input_flow = tf.placeholder(tf.float32, shape=(None, self.args.tw, self.imgsize, self.imgsize, 2))
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
        num_segs = (num_frames - self.args.tw) // self.args.stride + 1
        video_segs = np.zeros(shape=(num_segs, self.args.tw, self.imgsize, self.imgsize, 3))
        idx = 0
        for i in range(0, num_frames-self.args.tw, self.args.stride):
            video_segs[idx] = frames[i:i+self.args.tw, :, :, :]
            idx += 1
        feed_dict = {self.model_input_rgb: video_segs}
        model_out = self.sess.run([self.model_feature, self.model_logits, self.model_prediction], feed_dict)

        return model_out


    def extract_feature_flow(self, flows):
        num_frames = flows.shape[0]
        num_segs = (num_frames - self.args.tw) // self.args.stride + 1
        flow_segs = np.zeros(shape=(num_segs, self.args.tw, self.imgsize, self.imgsize, 2))
        idx = 0
        for i in range(0, num_frames-self.args.tw, self.args.stride):
            flow_segs[idx] = flows[i:i+self.args.tw, :, :, :]
            idx += 1
        feed_dict = {self.model_input_flow: flow_segs}
        model_out = self.sess.run([self.model_feature, self.model_logits, self.model_prediction], feed_dict)

        return model_out


    def run_all(self):
        for video in self.video_list: 
            if self.args.type == 'joint':
                video_path = os.path.join(self.args.video_dir, video)
                flow_path = os.path.join(self.args.flow_dir, video)
                if not os.path.isfile(video_path) or not os.path.isfile(flow_path):
                    print('File does not exist:', video_path)
                    continue
                frames = self.read_frames(video_path)
                flows = self.read_flows(flow_path)
                model_out = self.extract_feature_joint(frames, flows)
            elif self.args.type == 'rgb':
                video_path = os.path.join(self.args.video_dir, video)
                if not os.path.isfile(video_path):
                    print('File does not exist:', video_path)
                    continue
                frames = self.read_frames(video_path)
                model_out = self.extract_feature_rgb(frames)
            elif self.args.type == 'flow':
                flow_path = os.path.join(self.args.flow_dir, video)
                if not os.path.isfile(flow_path):
                    print('File does not exist:', flow_path)
                    continue
                flows = self.read_flows(flow_path)
                model_out = self.extract_feature_flow(flows)

            print(model_out[0].shape, model_out[1].shape, model_out[2].shape)
            print(np.argmax(model_out[2], axis=1))

        # with open(os.path.join(self.data_dir, 'kinetics_feature.pk'), 'wb') as fid:
        #     pickle.dump(self.features, fid, protocol=pickle.HIGHEST_PROTOCOL)


    def __init__(self, args):
        self.args = args
        if os.path.isfile(self.args.video_list):
            self.video_list = [line.strip() for line in open(self.args.video_list) if line[0] != '#']
        else:
            self.video_list = [self.args.video_list.strip()]
        self.imgsize = 224
        self.num_classes = 400
        
        self.model_input_rgb = None
        self.model_input_flow = None
        self.model_logits = None
        self.model_feature = None
        self.model_prediction = None
        self.sess = tf.Session()
        self.load_i3d_model()
        # self.features = {}


def main(args):
    kinetics_feature = KineticsFeature(args)
    kinetics_feature.run_all()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kinetics feature extraction')
    parser.add_argument('--video-dir', action='store')
    parser.add_argument('--flow-dir', action='store')
    parser.add_argument('--video-list', action='store')
    parser.add_argument('--type', choices=['rgb', 'flow', 'joint'], action='store')
    parser.add_argument('--tw', default=16, type=int)
    parser.add_argument('--stride', default=8, type=int)
    targs = parser.parse_args()
    main(targs)