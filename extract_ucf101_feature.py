import os
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import i3d
import tensorflow_hub as hub
import random
import pickle
import imageio
imageio.plugins.ffmpeg.download()

import warnings
warnings.filterwarnings("ignore")


class KineticsFeature(object):

    def read_avideo(self, video_path):
        reader = imageio.get_reader(video_path)
        video_np = np.zeros(shape=(1000, 240, 320, 3), dtype=np.uint8)
        for i, im in enumerate(reader):
            video_np[i] = im
        reader.close()
        return video_np[:i]


    def random_crop(self, video_np):
        x, y = random.randint(0, 96), random.randint(0, 16)
        return video_np[:, y:y+self.imgsize, x:x+self.imgsize, :]


    def center_crop(self, video_np):
        x, y = 48, 8
        return video_np[:, y:y+self.imgsize, x:x+self.imgsize, :]


    def extract_feature_hub(self, video):
        model_input = np.expand_dims(video, axis=0)

        # Create the i3d model and get the action probabilities.
        with tf.Graph().as_default():
            i3d = hub.Module("https://tfhub.dev/deepmind/i3d-kinetics-400/1")
            input_placeholder = tf.placeholder(shape=(None, None, 224, 224, 3), dtype=tf.float32)
            logits = i3d(input_placeholder)
            probabilities = tf.nn.softmax(logits)
            with tf.train.MonitoredSession() as session:
                [ps] = session.run(probabilities, feed_dict={input_placeholder: model_input})

        print("Top 5 actions:", np.argsort(ps)[::-1][:5])


    def load_i3d_model(self):
        self.model_input = tf.placeholder(tf.float32, shape=(None, None, self.imgsize, self.imgsize, 3))
        with tf.variable_scope('RGB'):
            self.rgb_model = i3d.InceptionI3d(self.num_classes, spatial_squeeze=True, final_endpoint='Predictions')
            self.preds, self.end_points = self.rgb_model(self.model_input, is_training=False, dropout_keep_prob=1.0)

        rgb_variable_map = {}
        for variable in tf.global_variables():
            if variable.name.split('/')[0] == 'RGB':
                rgb_variable_map[variable.name.replace(':0', '')] = variable
        rgb_saver = tf.train.Saver(var_list=rgb_variable_map, reshape=True)
        rgb_saver.restore(self.sess, 'data/checkpoints/rgb_imagenet/model.ckpt')
        print('RGB checkpoint restored')


    def extract_feature_local(self, video):
        video = np.expand_dims(video, axis=0)
        video_segs = np.zeros(shape=(500, 16, 224, 224, 3))
        idx = 0
        for i in range(0, video.shape[1]-16, 16):
            video_segs[idx] = video[0, i:i+16, :, :]
            idx += 1
        video_segs = video_segs[:idx]
        out_preds, out_logits = self.sess.run([self.preds, self.end_points['Last_pool']], {self.model_input: video_segs})
        return out_preds, out_logits


    def __call__(self):
        for video in tqdm(self.video_list):
            video_path = os.path.join(self.data_dir, 'videos', video.split()[0])
            if not os.path.exists(video_path):
                print('File does not exist:', video_path)
                exit(0)

            try:
                video_np = self.read_avideo(video_path)
                video_np = self.center_crop(video_np)
                self.features[video.split()[0]] = self.extract_feature_local(video_np)
            except:
                pass

        with open(os.path.join(self.data_dir, 'kinetics_feature.pk'), 'wb') as fid:
            pickle.dump(self.features, fid, protocol=pickle.HIGHEST_PROTOCOL)



    def __init__(self, data_dir, video_list_path):
        self.data_dir = data_dir
        self.video_list = [line.strip() for line in open(video_list_path)]
        self.imgsize = 224
        self.num_classes = 400
        self.sess = tf.Session()
        self.load_i3d_model()
        self.features = {}


def main():
    data_dir = '/mnt/research-6f/mhasan/data/ucf-101'
    video_list_path = '/mnt/research-6f/mhasan/data/ucf-101/allvids.txt'
    kinetics_feature = KineticsFeature(data_dir, video_list_path)
    kinetics_feature()


if __name__ == '__main__':
    main()
