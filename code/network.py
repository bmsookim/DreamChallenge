import tensorflow as tf
import numpy as np
import config as cf
import functions as F
import os
import csv
from PIL import Image
from numpy import array
import augmentation as aug
from tensorflow.python.framework import ops
from functions import *
from config import *

class BasicConvNet(object):
    def __init__(self, image_w=cf.w, image_h=cf.h, channels=cf.channels, num_classes=2):
        self._width  = image_w # define the width of the image.
        self._height = image_h # define the height of the image.
        self._batch_size = cf.batch_size # define the batch size of mini-batch training.
        self._channels = cf.channels # define the number of channels. ex) RGB = 3, GrayScale = 1, FeatureMap = 50
        self._num_classes = num_classes # define the number of classes for final classfication

        # define the basic options for tensorflow session : restricts allocation of GPU memory.
        gpu_options = tf.GPUOptions(allow_growth = True)
        self._session = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, allow_soft_placement=True))

        # placeholders : None will become the batch size of each batch. The last batch of an epoch may be volatile.
        self._CC = tf.placeholder(tf.float32, shape=[None, self._width, self._height, self._channels])
        self._MLO = tf.placeholder(tf.float32, shape=[None, self._width, self._height, self._channels])
        self._labels = tf.placeholder(tf.int64, shape=[None])
        self._keep_prob = tf.placeholder(tf.float32)
        self._is_train = tf.placeholder(tf.bool)
        self._global_step = tf.Variable(0, tf.int64, name="global_step") # saves the global step of training.

        # loss calculation & update
        self._logits = self._inference(self._CC, self._MLO, self._keep_prob, self._is_train) # prediction
        self._avg_loss = self._loss(self._labels, self._logits) # difference between prediction & actual label.
        self._train_op = self._train(self._avg_loss) # back propagate the loss.
        self._accuracy = F.accuracy_score(self._labels, self._logits) # get the accuracy of given prediction batch.

        # basic tensorflow run operations
        self._saver = tf.train.Saver(tf.all_variables())
        self._session.run(tf.initialize_all_variables())

    def fit(self):
        patient_lst = os.listdir(data_dir)
        cnt = 0
        cancer_cnt = 0

        meta_dict = os.path.join(data_dir, 'metadata.tsv')

        for patient in patient_lst:
            if patient != 'metadata.tsv':
                exam_lst = os.listdir(data_dir + patient)
                for exam in exam_lst :
                    lat_lst = os.listdir(os.path.join(data_dir, patient, exam))
                    for lat in lat_lst:
                        view_lst = os.listdir(os.path.join(data_dir, patient, exam, lat))
                        img_bin = []
                        img_label = []
                        tsv_in = open(meta_dict, 'rb')
                        tsv_in = csv.reader(tsv_in, delimiter = '\t')
                        for (i, e, v, l, cancer) in tsv_in:
                            if(i == patient):
                               if(e == exam):
                                   if(l == lat):
                                      img_label = []
                                      img_label.append(int(cancer))
                                      img_label = np.reshape(img_label, (1,))
                        for view in view_lst:
                            img = Image.open(os.path.join(data_dir, patient, exam, lat, view))
                            img = array(img)
                            img = np.reshape(img, (1, img.shape[0], img.shape[1], cf.channels))
                            img_bin.append(img)
                            cnt = cnt+1


                        # feeding the dictionary for every placeholder
                        feed_dict = {
                            self._CC : img_bin[0],
                            self._MLO : img_bin[1],
                            self._labels : img_label,
                            self._is_train : True,
                            self._keep_prob : cf.keep_prob
                        }

                        # fetching the outputs for session
                        _, acc, train_avg_loss, global_step = self._session.run(
                            fetches = [
                                self._train_op,
                                self._accuracy,
                                self._avg_loss,
                                self._global_step],
                            feed_dict = feed_dict)

                        if(img_label == 1):
                            cancer_cnt = cancer_cnt + 1 
                            print('\nCancer detected #%d' %cancer_cnt)
                            print('Minibatch loss at batch %d: %.5f' %((cnt), train_avg_loss))
                            print('Minibatch accuracy: %.2f%%' %(acc*100))

        print("\nTotal number of examples : " + str(cnt))

    def save(self, filepath):
        self._saver.save(self._session, filepath)
    def load(self, filepath):
        self._save.restore(self._session, filepath)
        print("Model restored")

    def _inference(self, CC, MLO, keep_prob, is_train):
        pass

    def _loss(self, labels, logits):
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits, labels)
        cross_entropy_mean = tf.reduce_mean(cross_entropy)
        tf.add_to_collection('losses', cross_entropy_mean)
        entropy_losses = tf.get_collection('losses')
        regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        loss_ = tf.add_n(entropy_losses + regularization_losses)
        return loss_

    def _train(self, avg_loss):
        return tf.train.AdamOptimizer(learning_rate=0.001).minimize(avg_loss)


class challenge_net (BasicConvNet):
    def _inference(self, CC, MLO, keep_prob, is_train):
        layers = [3, 16, 32, 64, 64]
        
        cc = CC
        mlo = MLO

        for i in range(4):
            with tf.variable_scope('CC_layers_%s' %i) as scope:
                cc = F.conv(cc, layers[i])
                cc = F.batch_norm(cc, is_train)
                cc = F.activation(cc)
            cc = F.max_pool(cc)
        with tf.variable_scope('CC_features') as scope:
            cc = F.dense(cc, layers[i+1])
            cc = F.batch_norm(cc, is_train)
            cc = F.activation(cc)
        
        for j in range(4):
            with tf.variable_scope('MLO_layers_%s' %j) as scope:
                mlo = F.conv(mlo, layers[j])
                mlo = F.batch_norm(mlo, is_train)
                mlo = F.activation(mlo)
            mlo = F.max_pool(mlo)
        with tf.variable_scope('MLO_features') as scope:
            mlo = F.dense(mlo, layers[j+1])
            mlo = F.batch_norm(mlo, is_train)
            mlo = F.activation(mlo)

        with tf.variable_scope('softmax') as scope:
            concat = tf.concat(1, [cc, mlo])
            h = F.dense(concat, self._num_classes)

        return h

class vggnet(BasicConvNet):
    def _inference(self, CC, MLO, keep_prob, is_train):
        dropout_rate = [0.9, 0.8, 0.7, 0.6, 0.5]
        layers = [64, 128, 256, 512, 512]
        iters = [2, 2, 3, 3]
        h = X

        # VGG Network Layer
        for i in range(4):
            for j in range(iters[i]):
                with tf.variable_scope('layers%s_%s' %(i, j)) as scope:
                    h = F.conv(h, layers[i])
                    h = F.batch_norm(h, is_train)
                    h = F.activation(h)
                    h = F.dropout(h, dropout_rate[i], is_train)
            h = F.max_pool(h)

        # Fully Connected Layer
        with tf.variable_scope('fully_connected_layer') as scope:
            h = F.dense(h, layers[i+1])
            h = F.batch_norm(h, is_train)
            h = F.activation(h)
            h = F.dropout(h, dropout_rate[i+1], is_train)

        # Softmax Layer
        with tf.variable_scope('softmax_layer') as scope:
            h = F.dense(h, self._num_classes)

        return h
