# -*- coding: utf-8 -*-
# @Time        : 16/1/2019 5:49 PM
# @Description :
# @Author      : li rui hui
# @Email       : ruihuili@gmail.com
import tensorflow as tf
from Common import ops
from tf_ops.sampling.tf_sampling import gather_point, farthest_point_sample

class Twin(object):
    def __init__(self, opts,is_training, name="twin"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.out_num_point = int(self.num_point*self.up_ratio)

    def __call__(self, inputs, outputs):
        with tf.variable_scope(self.name, reuse=self.reuse):
            input = tf.expand_dims(inputs,axis = 2)
            output = tf.expand_dims(outputs, axis=2)
            input = ops.conv2d(input,16, [1, 1], padding='VALID', stride=[1, 1], bn=False,
                               is_training=self.is_training, scope='layer_0', bn_decay=None)
            input = ops.conv2d(input, 64, [1, 1], padding='VALID', stride=[1, 1], bn=False,
                               is_training=self.is_training, scope='layer_1', bn_decay=None)
            output = ops.conv2d(output, 16, [1, 1], padding='VALID', stride=[1, 1], bn=False,
                               is_training=self.is_training, scope='layer_0', bn_decay=None)
            output = ops.conv2d(output, 64, [1, 1], padding='VALID', stride=[1, 1], bn=False,
                               is_training=self.is_training, scope='layer_1', bn_decay=None)
            input = ops.SE_NET(input, scope='channel', is_training=self.is_training)
            output = ops.SE_NET(output, scope='channel', is_training=self.is_training)
            input = tf.squeeze(input ,axis = 2)
            output = tf.squeeze(output,axis = 2)

        return input, output


class Generator(object):
    def __init__(self, opts,is_training, name="Generator"):
        self.opts = opts
        self.is_training = is_training
        self.name = name
        self.reuse = False
        self.num_point = self.opts.patch_num_point
        self.up_ratio = self.opts.up_ratio
        self.up_ratio_real = self.up_ratio + 2
        self.out_num_point = int(self.num_point*self.up_ratio)

    def __call__(self, inputs):
        with tf.variable_scope(self.name, reuse=self.reuse):

            features, ori_feature = ops.feature_extraction_RCB(inputs, scope='feature_extraction', is_training=self.is_training, bn_decay=None)
            # downsampling for points and features
            idx = farthest_point_sample(self.num_point, inputs)
            idx_1 = idx[:, 0:int(self.num_point/2)]
            idx_2 = idx[:, int(self.num_point/2):self.num_point]
            point_1 = gather_point(inputs, idx_1)
            point_2 = gather_point(inputs, idx_2)
            feature_1 = ops.gather_features(features, idx_1)
            feature_2 = ops.gather_features(features, idx_2)
            feature_1 = tf.concat([feature_1,point_1],axis = -1)
            feature_2 = tf.concat([feature_2,point_2],axis = -1)
            ori_feature_1 = ops.gather_features(ori_feature, idx_1)
            ori_feature_2 = ops.gather_features(ori_feature, idx_2)
            # train self similarity model  (SSM)
            H1 = ops.Self_Similarity_Model(inputs, feature_1, int(self.up_ratio/2), scope="Self_Model_1",
                                           is_training=self.is_training, bn_decay=None) # [B,0.5N,1,2C]
            L1 = tf.reshape(H1, [tf.shape(inputs)[0], tf.shape(inputs)[1], 1, -1]) # [B,N,1,C] ]eg,[1111,2222,3333]
            # predictied point via SSM
            Pred_point_1 = ops.Coordinate_generator(L1, scope = 'Coord_generator', is_training=True)
            # use SSM
            H2 = ops.Self_Similarity_Model(inputs, feature_2, int(self.up_ratio/2), scope="Self_Model_1",
                                           is_training=self.is_training, bn_decay=None) # [B,0.5N,1,2C]
            H = tf.concat([H1, H2], axis=1)  # [B,N,1,2C]
            F = tf.expand_dims(tf.concat([ori_feature_1,ori_feature_2],axis = 1), axis=2) # [B,N,1,2C']
            F = ops.conv2d(F, 512, [1, 1], padding='VALID', stride=[1, 1], bn=False, is_training=self.is_training,
                           scope='align_layer', bn_decay=None)
            # residual
            H = H - F
            H = ops.conv2d(H, 3072, [1, 1],padding='VALID', stride=[1, 1],bn=False, is_training=self.is_training,
                               scope='up_layer', bn_decay=None)
            H = tf.reshape(H, [tf.shape(inputs)[0], tf.shape(inputs)[1]*self.up_ratio_real, 1, -1])  #[B,rN,1,C]
            coord = ops.conv2d(H, 64, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer1', bn_decay=None)
            coord = ops.conv2d(coord, 3, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=self.is_training,
                               scope='fc_layer2', bn_decay=None,
                               activation_fn=None, weight_decay=0.0)
            P_order = tf.concat([point_1,point_2],axis = 1) # [B,N,3]
            ori = tf.tile(P_order, [1, 1, self.up_ratio_real])
            ori = tf.reshape(ori, [tf.shape(P_order)[0], -1, 3])
            ori = tf.expand_dims(ori ,axis = 2)
            out = ori + coord
            offset = ops.geometry_refiner(out, H, scope='refine', is_training=self.is_training)
            outputs = out + offset
            outputs = tf.squeeze(outputs, [2])

            outputs = gather_point(outputs, farthest_point_sample(self.out_num_point, outputs))

        self.reuse = True
        self.variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
        return outputs, Pred_point_1