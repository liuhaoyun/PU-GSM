# Author: Wentao Yuan (wyuan1@cs.cmu.edu) 05/31/2018

import tensorflow as tf
import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.getcwd()))
def mlp(features, layer_dims, bn=None, bn_params=None):
    for i, num_outputs in enumerate(layer_dims[:-1]):
        features = tf.contrib.layers.fully_connected(
            features, num_outputs,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='fc_%d' % i)
    outputs = tf.contrib.layers.fully_connected(
        features, layer_dims[-1],
        activation_fn=None,
        scope='fc_%d' % (len(layer_dims) - 1))
    return outputs


def mlp_conv(inputs, layer_dims, bn=None, bn_params=None):
    for i, num_out_channel in enumerate(layer_dims[:-1]):
        inputs = tf.contrib.layers.conv2d(
            inputs, num_out_channel,
            kernel_size=1,
            normalizer_fn=bn,
            normalizer_params=bn_params,
            scope='conv_%d' % i)
    outputs = tf.contrib.layers.conv2d(
        inputs, layer_dims[-1],
        kernel_size=1,
        activation_fn=None,
        scope='conv_%d' % (len(layer_dims) - 1))
    return outputs

##################################################################################
# Back projection Blocks
##################################################################################
def PointShuffler(inputs, scale=2):
    #inputs: B x N x 1 X C
    #outputs: B x N*scale x 1 x C//scale
    outputs = tf.reshape(inputs,[tf.shape(inputs)[0],tf.shape(inputs)[1],1,tf.shape(inputs)[3]//scale,scale])
    outputs = tf.transpose(outputs,[0, 1, 4, 3, 2])

    outputs = tf.reshape(outputs,[tf.shape(inputs)[0],tf.shape(inputs)[1]*scale,1,tf.shape(inputs)[3]//scale])

    return outputs

from Common.model_utils import gen_1d_grid,gen_grid
def up_block(inputs, up_ratio, scope='up_block', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        dim = inputs.get_shape()[-1]
        out_dim = dim*up_ratio
        grid = gen_grid(up_ratio)

        grid = tf.tile(tf.expand_dims(grid, 0), [tf.shape(net)[0], 1,tf.shape(net)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(net)[0], -1, 1, 2])
            #grid = tf.expand_dims(grid, axis=2)

        net = tf.tile(net, [1, up_ratio, 1, 1])
        net = tf.concat([net, grid], axis=-1)

        net = attention_unit(net, is_training=is_training)

        net = conv2d(net, 256, [1, 1],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net

def down_block(inputs,up_ratio,scope='down_block',is_training=True,bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        net = inputs
        net = tf.reshape(net,[tf.shape(net)[0],up_ratio,-1,tf.shape(net)[-1]])
        net = tf.transpose(net, [0, 2, 1, 3])

        net = conv2d(net, 256, [1, up_ratio],
                                 padding='VALID', stride=[1, 1],
                                 bn=False, is_training=is_training,
                                 scope='conv1', bn_decay=bn_decay)
        net = conv2d(net, 128, [1, 1],
                          padding='VALID', stride=[1, 1],
                          bn=False, is_training=is_training,
                          scope='conv2', bn_decay=bn_decay)

    return net

def Chain_Residual_Block(input, output=128, block_num=4, scope='chain_residual_block',
                         is_training=True,  use_bn=False, use_ibn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        identity = input
        sum_residual = identity  # 存储每个链的残差

        for i in range(block_num):
            if i == 0:
                feature, residual = Rssidual_Block(identity, output, scope='residual_block%d' % i, is_training=is_training,
                                             use_bn = use_bn, use_ibn = use_ibn, bn_decay=bn_decay)
                sum_residual = residual
            else:
                feature, residual = Rssidual_Block(feature, output, scope='residual_block%d' % i, is_training=is_training,
                                             use_bn = use_bn, use_ibn = use_ibn, bn_decay=bn_decay)
                sum_residual = tf.concat([sum_residual, residual], axis=-1)  # concat 所有链的 residuals  [n * 128]

        sum_residual = SE_NET(sum_residual, scope='se_net',is_training=is_training)  # 对残差进行 attention
        sum_residual = conv2d(sum_residual, output, [1, 1],
                              padding='VALID', scope='layer_compress', is_training=is_training, bn=use_bn, ibn=use_ibn,
                              bn_decay=bn_decay, activation_fn=None)  # [128]  No Relu
        out = identity + sum_residual
        out = tf.nn.relu(out)
    return out

def RCB_conv(input, k, output=256, block_num=3, layer=1, scope='CRB', is_training=True,
             use_bn=False, use_ibn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        y, idx = get_edge_feature(input, k=k, idx=None)  # [B N K 2*C]
        for i in range(layer):
            y = Chain_Residual_Block(y, output, block_num, scope='chain_residual_block%d' % i,
                                     is_training=is_training, use_bn=use_bn, use_ibn=use_ibn, bn_decay=bn_decay)  # [B, N, 128]

        y = conv2d(y, output, [1, 1], padding='VALID', scope='Adjust_layer', activation_fn = None,
                   is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # 使用 Conv without Relu

        # attentive_pooling
        #y = att_pooling(y, scope='attentive_pooling') # [B, N, C]
        # max_pooling

        y = tf.reduce_max(y, axis=-2)  # [B, N, C]

        return y, idx

##################################################################################
                           # gather_features
##################################################################################
# input : [B,N,C],
# idx : [B,k]
# output : [B,N,k]

def gather_features(input, idx):
    B, N, C = [i.value for i in input.get_shape()]
    _, k = [j.value for j in idx.get_shape()]

    idx_bais = tf.reshape(tf.range(0, B), [B, 1]) * N
    idx_bais_tile = tf.tile(idx_bais, [1, k])
    index_new = idx + idx_bais_tile  # 接上索引信息
    input_reshape = tf.reshape(input, [B * N, -1])  # 把输入由 原来的 [B,N,C] 变成 [B*N, C]
    new_point = tf.gather_nd(input_reshape, tf.reshape(index_new, [B * k, -1]))
    new_point = tf.reshape(new_point, [B, k, -1])  # [B,N,C] 输出经过高通图滤波器采样后的点云

    return new_point


def feature_extraction_RCB(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):
    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):
        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 17
        comp = growth_rate
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, growth_rate, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = RCB_conv(l0_features, k=knn, output= growth_rate * 2, scope="layer1",
                                       is_training=is_training,bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # 96

        l2_features = conv1d(l1_features, comp*2, 1,  padding='VALID', scope='layer2_prep',
                             is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)

        l2_features, l2_idx  = RCB_conv(l2_features, k=knn, output= growth_rate * 4, scope="layer2",
                                       is_training=is_training,bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 224

        l3_features = conv1d(l2_features, comp*3, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = RCB_conv(l3_features, k=knn, output= growth_rate * 6, scope="layer3",
                                       is_training=is_training,bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 352

        l4_features = conv1d(l3_features, comp*3, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l4_idx = RCB_conv(l4_features, k=knn, output= growth_rate * 6, scope="layer4",
                                       is_training=is_training,bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 480

    return l4_features, l0_features

def feature_extraction(inputs, scope='feature_extraction2', is_training=True, bn_decay=None):

    with tf.variable_scope(scope,reuse=tf.AUTO_REUSE):

        use_bn = False
        use_ibn = False
        growth_rate = 24

        dense_n = 3
        knn = 16
        comp = growth_rate*2
        l0_features = tf.expand_dims(inputs, axis=2)
        l0_features = conv2d(l0_features, 24, [1, 1],
                                     padding='VALID', scope='layer0', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay, activation_fn=None)
        l0_features = tf.squeeze(l0_features, axis=2)

        # encoding layer
        l1_features, l1_idx = dense_conv(l0_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer1", is_training=is_training, bn=use_bn, ibn=use_ibn,
                                                  bn_decay=bn_decay)
        l1_features = tf.concat([l1_features, l0_features], axis=-1)  # (12+24*2)+24=84

        l2_features = conv1d(l1_features, comp, 1,  # 24
                                     padding='VALID', scope='layer2_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)
        l2_features, l2_idx = dense_conv(l2_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer2", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l2_features = tf.concat([l2_features, l1_features], axis=-1)  # 84+(24*2+12)=144

        l3_features = conv1d(l2_features, comp, 1,  # 48
                                     padding='VALID', scope='layer3_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l3_features, l3_idx = dense_conv(l3_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer3", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l3_features = tf.concat([l3_features, l2_features], axis=-1)  # 144+(24*2+12)=204

        l4_features = conv1d(l3_features, comp, 1,  # 48
                                     padding='VALID', scope='layer4_prep', is_training=is_training, bn=use_bn, ibn=use_ibn,
                                     bn_decay=bn_decay)  # 48
        l4_features, l3_idx = dense_conv(l4_features, growth_rate=growth_rate, n=dense_n, k=knn,
                                                  scope="layer4", is_training=is_training, bn=use_bn, bn_decay=bn_decay)
        l4_features = tf.concat([l4_features, l3_features], axis=-1)  # 204+(24*2+12)=264

        l4_features = tf.expand_dims(l4_features, axis=2)

    return l4_features


# input :  (B, N, 1, C)
# output : (B, N ,1, C)
def Rssidual_Block(input, C_OUT, scope='residual_block', is_training=True,
                   use_ibn=False, use_bn=False, bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gamma = 4  # bottleNeck ratio
        x = input
        # RB : Conv + Bn + Relu
        residual = conv2d(input, C_OUT//gamma, [1, 1],
                     padding='VALID', scope='bottle_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                     bn_decay=bn_decay)

        residual = conv2d(residual, C_OUT, [1, 1],
                          padding='VALID', scope='bottle_2', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)  # Conv + bn ， 最后一层 没有 Relu

        y = x + residual
        y = tf.nn.relu(y)
    return y, residual

def Self_Similarity_Model(xyz, input, up_ratio, output=512, scope='Self_Similarity_Model',
                               is_training=True,  use_bn=False, use_ibn=False, bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        grid_dim = 2
        grid = gen_grid(up_ratio)
        grid = tf.tile(tf.expand_dims(grid, 0),[tf.shape(input)[0], 1, tf.shape(input)[1]])  # [batch_size, num_point*4, 2])
        grid = tf.reshape(grid, [tf.shape(input)[0], -1, 1, grid_dim])
        grid0 = grid[:, 0:tf.shape(input)[1], :, :]
        inputs = tf.expand_dims(input, axis=2)
        net = tf.concat([inputs, grid0], axis = -1)
        # align feature
        net = conv2d(net, 256, [1, 1], padding='VALID', scope='Adjust_layer',
                         is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # 使用 Conv + Relu
        # Encoder
        net = attention_unit(net, scope='Encoder_L0_Head0',is_training=is_training)
        # Multi-head Transformer Decoder

        L0H0 = attention_unit(net, scope='Decoder_L0_Head0', is_training=is_training)
        L0H1 = attention_unit(net, scope='Decoder_L0_Head1', is_training=is_training)
        L0 = tf.concat([L0H0,L0H1],axis = -1)
        L0 = conv2d(L0, output, [1, 1], padding='VALID', scope='MLP_Layer2',
                            is_training=is_training, bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)  # 使用 Conv + Relu
    return L0


def Coordinate_generator(features, out_dim = 64, scope = 'coord_generator', is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):

        feature = conv2d(features, out_dim, [1, 1], padding='VALID', scope='MLP_0', is_training=is_training)
        point = conv2d(feature, 3, [1, 1], padding='VALID', scope='MLP_1', is_training=is_training, activation_fn=None)
        point = tf.squeeze(point, [2]) #[B,N,C]

    return point

def geometry_refiner(xyz_up, input_res, out_dim = 128, layer = 2, knn = 26, scope = 'Coordinate_refine', is_training=True):
   with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
       # geometry knn
       xyz = tf.squeeze(xyz_up, [2])
       knn_xyz, idx = get_KNN_feature(xyz, k=knn)  # [B,N,K,3]
       central_xyz = tf.expand_dims(xyz, axis = 2) # [B,N,1,3]
       central_xyz = tf.tile(central_xyz, [1, 1, knn, 1]) # [B,N,K,3]
       local_res = central_xyz-knn_xyz
       global_res = conv2d(input_res, out_dim //4, [1, 1], padding='VALID', scope='MLP_1',
                           is_training=is_training)  # Conv + Relu
       g_res = tf.tile(global_res, [1, 1, knn, 1])  # [B,N,K,3]
       res = tf.concat([central_xyz, local_res, g_res], axis = -1) # [B,N,K,6]
       res = conv2d(res, out_dim//4, [1, 1], padding='VALID', scope='MLP_2', is_training=is_training,
                         activation_fn=None)  # Conv + Relu
       for i in range(layer):
           res, residual = Rssidual_Block(res, out_dim//4, scope='residual_layer%d' % i)

       res = tf.reduce_max(res, axis=2, keep_dims=True)
       coord = conv2d(res, 16, [1, 1],
                      padding='VALID', stride=[1, 1],
                      bn=False, is_training=is_training,
                      scope='fc_layer1', bn_decay=None)

       coord = conv2d(coord, 3, [1, 1],
                      padding='VALID', stride=[1, 1],
                      bn=False, is_training=is_training,
                      scope='fc_layer2', bn_decay=None,
                      activation_fn=None, weight_decay=0.0)

       return coord

def Coordinate_Refine(xyz_up, features, out_dim = 128, layer = 1, knn = 26, scope = 'Coordinate_refine', is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # geometry knn
        xyz = tf.squeeze(xyz_up, [2])
        knn_xyz, idx = get_KNN_feature(xyz, k=knn)  # [B,N,K,3]
        central_xyz = tf.expand_dims(xyz, axis = 2) # [B,N,1,3]
        central_xyz = tf.tile(central_xyz, [1, 1, knn, 1]) # [B,N,K,3]
        edge_xyz = tf.concat([central_xyz, central_xyz-knn_xyz], axis = -1)
        edge_xyz = conv2d(edge_xyz, out_dim//4, [1, 1], padding='VALID', scope='MLP_1',is_training=is_training)  # 使用 Conv + Relu
        edge_xyz = conv2d(edge_xyz, out_dim//2, [1, 1], padding='VALID', scope='MLP_2', is_training=is_training,activation_fn=None)  # 使用 Conv + Relu
        for i in range(layer):
            edge_xyz, residual = Rssidual_Block(edge_xyz, out_dim//2, scope='residual_layer%d' % i)
        edge_xyz = conv2d(edge_xyz, out_dim, [1, 1], padding='VALID', scope='MLP_3', is_training=is_training,activation_fn=None)
        feature = conv2d(features, out_dim, [1, 1], padding='VALID', scope='MLP_4', is_training=is_training,activation_fn=None)
        feature =  tf.tile(feature, [1, 1, knn, 1]) # [B,N,K,3]
        res = edge_xyz - feature
        res = tf.reduce_max(res, axis=2, keep_dims=True)
        coord = conv2d(res, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer2', bn_decay=None,
                           activation_fn=None, weight_decay=0.0)

        return coord

def up_projection_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        L = conv2d(inputs, 128, [1, 1],
                               padding='VALID', stride=[1, 1],
                               bn=False, is_training=is_training,
                               scope='conv0', bn_decay=bn_decay)

        H0 = up_block(L,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_0')

        L0 = down_block(H0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='down_0')
        E0 = L0-L
        H1 = up_block(E0,up_ratio,is_training=is_training,bn_decay=bn_decay,scope='up_1')
        H2 = H0+H1
    return H2

def weight_learning_unit(inputs,up_ratio,scope="up_projection_unit",is_training=True,bn_decay=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape().as_list()[-1]
        grid = gen_1d_grid(tf.reshape(up_ratio,[]))

        out_dim = dim * up_ratio

        ratios = tf.tile(tf.expand_dims(up_ratio,0),[1,tf.shape(grid)[1]])
        grid_ratios = tf.concat([grid,tf.cast(ratios,tf.float32)],axis=1)
        weights = tf.tile(tf.expand_dims(tf.expand_dims(grid_ratios,0),0),[tf.shape(inputs)[0],tf.shape(inputs)[1], 1, 1])
        weights.set_shape([None, None, None, 2])
        weights = conv2d(weights, dim, [1, 1],
                   padding='VALID', stride=[1, 1],
                   bn=False, is_training=is_training,
                   scope='conv_1', bn_decay=None)


        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_2', bn_decay=None)
        weights = conv2d(weights, out_dim, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=False, is_training=is_training,
                         scope='conv_3', bn_decay=None)

        s = tf.matmul(hw_flatten(inputs), hw_flatten(weights), transpose_b=True)  # # [bs, N, N]

    return tf.expand_dims(s,axis=2)


def coordinate_reconstruction_unit(inputs,scope="reconstruction",is_training=True,bn_decay=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        coord = conv2d(inputs, 64, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer1', bn_decay=None)

        coord = conv2d(coord, 3, [1, 1],
                           padding='VALID', stride=[1, 1],
                           bn=False, is_training=is_training,
                           scope='fc_layer2', bn_decay=None,
                           activation_fn=None, weight_decay=0.0)
        outputs = tf.squeeze(coord, [2])

        return outputs

def SE_NET_PLUS(input, scope='se_net', is_training=True, bn_decay=None, use_bn=False, use_ibn=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        c = input.get_shape().as_list()[3]
        feature = tf.reduce_mean(input, axis=[1, 2])  # Global Average Pooling [B C]
        feature = tf.expand_dims(tf.expand_dims(feature, axis=1), axis=2) # [B 1 1 C]
        DC = tf.tile(feature, [1, input.get_shape().as_list()[1], 1, 1])  # [B,N,1,C]
        Res = input - DC
        Global = conv2d(Res, 1, [1, 1], padding='VALID', scope='global_0', is_training=is_training,
                        bn=use_bn, ibn=use_ibn, bn_decay=bn_decay, activation_fn=None)
        Global = tf.nn.softmax(Global, axis=1)  # [B,N,1,1]
        Res = tf.matmul(hw_flatten(Global), hw_flatten(Res), transpose_a=True)  # # [B, 1, C]
        Res = tf.expand_dims(Res, axis=1)  # [B 1 1 C]
        feature = tf.concat([feature, Res], axis = -1) # [B,1,1,2C]
        feature = conv2d(feature, c // 16, [1, 1], padding='VALID', scope='DC_0', is_training=is_training,
                         bn=use_bn, ibn=use_ibn, bn_decay=bn_decay)
        feature = conv2d(feature, c, [1, 1], padding='VALID', scope='DC_1', is_training=is_training,
                         bn=use_bn, ibn=use_ibn, bn_decay=bn_decay, activation_fn=None)
        feature = feature
        scale = tf.sigmoid(feature)  # [B,1,1,C]
        out = input * scale
    return out

def GC_NET(input, scope='GC_net', is_training=True, bn_decay=None, use_bn = False, use_ibn=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        c = input.get_shape().as_list()[3]
        Global = conv2d(input, 1, [1, 1],padding='VALID', scope='SE_0', is_training=is_training,
                         bn=use_bn, ibn=use_ibn, bn_decay=bn_decay, activation_fn=None)
        Global = tf.nn.softmax(Global, axis = 1) # [B,N,1,1]
        feature = tf.matmul(hw_flatten(Global), hw_flatten(input), transpose_a=True)  # # [bs, 1, 1, C]
        feature = tf.expand_dims(feature, axis=1)  # [B 1 1 C]
        feature = conv2d(feature, c//16, [1, 1],
                          padding='VALID', scope='SE_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay)
        feature = conv2d(feature, c, [1, 1],
                          padding='VALID', scope='SE_2', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)

        scale = tf.sigmoid(feature)  # [B,1,1,C]
        out = input * scale
    return out

# input ： (B, N, 1，C)
# output : (B, N ,1, C)
def SE_NET(input, scope='se_net', is_training=True, bn_decay=None, use_bn = False, use_ibn=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        B, N, K, C = [i.value for i in input.get_shape()]
        mean = tf.reduce_mean(input, axis=[1, 2])  # Global Average Pooling [B C]
        mean = tf.expand_dims(tf.expand_dims(mean, axis=1), axis=2) # [B 1 1 C]
        mean_knn = tf.tile(mean, [1,N,K,1]) # [B,N,K,C]
        res = input - mean_knn
        res = conv2d(res, C, [1, 1],padding='VALID', scope='SE_0',is_training=is_training,
                     bn=use_bn, ibn=use_ibn, bn_decay=bn_decay, activation_fn=None)
        res = tf.reduce_max(res, axis=[1, 2])  # Global Average Pooling [B C]
        res = tf.expand_dims(tf.expand_dims(res, axis=1), axis=2)  # [B 1 1 C]
        feature = mean + res
        feature = conv2d(feature, C//16, [1, 1],
                          padding='VALID', scope='SE_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay)
        feature = conv2d(feature, C, [1, 1],
                          padding='VALID', scope='SE_2', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)

        scale = tf.sigmoid(feature)  # [B,1,1,C]
        out = input * scale
    return out

# SE-Net (Channel attention model)
# input ： (B, N, 1，C)
# output : (B, N ,1, C)
def SE_NET_0(input, scope='se_net', is_training=True, bn_decay=None, use_bn = False, use_ibn=False):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        c = input.get_shape().as_list()[3]
        feature = tf.reduce_mean(input, axis=[1, 2])  # Global Average Pooling [B C]
        feature = tf.expand_dims(feature, axis=1)  # [B 1 C]
        feature = tf.expand_dims(feature, axis=2)  # [B 1 1 C]
        #feature = conv2d(feature, c//16, [1, 1],
        #                  padding='VALID', scope='SE_0', is_training=is_training, bn=use_bn, ibn=use_ibn,
        #                  bn_decay=bn_decay)
        feature = conv2d(feature, c, [1, 1],
                          padding='VALID', scope='SE_1', is_training=is_training, bn=use_bn, ibn=use_ibn,
                          bn_decay=bn_decay, activation_fn=None)

        scale = tf.sigmoid(feature)  # [B,1,1,C]
        out = input * scale
    return out


# non-local attention (self-attention)
def transformer_unit(Q,K,V, scope='transformer_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = V.get_shape()[-1].value
        layer = dim//4
        f = conv2d(Q,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = conv2d(K, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = conv2d(V, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)

        # channel attention
        h = SE_NET(h, scope='Self_A', is_training=is_training)  # 对残差进行 attention

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=V.shape)  # [bs, h, w, C]
        x = gamma * o + V

    return x

def attention_unit(inputs, scope='attention_unit',is_training=True):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        dim = inputs.get_shape()[-1].value
        layer = dim//4
        f = conv2d(inputs,layer, [1, 1],
                              padding='VALID', stride=[1, 1],
                              bn=False, is_training=is_training,
                              scope='conv_f', bn_decay=None)

        g = conv2d(inputs, layer, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_g', bn_decay=None)

        h = conv2d(inputs, dim, [1, 1],
                            padding='VALID', stride=[1, 1],
                            bn=False, is_training=is_training,
                            scope='conv_h', bn_decay=None)

        # channel attention
        h = SE_NET(h, scope='GCSE', is_training=is_training)  # 对残差进行 attention

        s = tf.matmul(hw_flatten(g), hw_flatten(f), transpose_b=True)  # # [bs, N, N]

        beta = tf.nn.softmax(s, axis=-1)  # attention map

        o = tf.matmul(beta, hw_flatten(h))   # [bs, N, N]*[bs, N, c]->[bs, N, c]
        gamma = tf.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))

        o = tf.reshape(o, shape=inputs.shape)  # [bs, h, w, C]
        x = gamma * o + inputs

    return x


##################################################################################
# Other function
##################################################################################
def instance_norm(net, train=True,weight_decay=0.00001):
    batch, rows, cols, channels = [i.value for i in net.get_shape()]
    var_shape = [channels]
    mu, sigma_sq = tf.nn.moments(net, [1, 2], keep_dims=True)

    shift = tf.get_variable('shift',shape=var_shape,
                            initializer=tf.zeros_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    scale = tf.get_variable('scale', shape=var_shape,
                            initializer=tf.ones_initializer,
                            regularizer=tf.contrib.layers.l2_regularizer(weight_decay))
    epsilon = 1e-3
    normalized = (net - mu) / tf.square(sigma_sq + epsilon)
    return scale * normalized + shift


def conv1d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=1,
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn=False,
           bn_decay=None,
           use_bias=True,
           is_training=None,
           reuse=None):
    """ 1D convolution with non-linear operation.

    Args:
        inputs: 3-D tensor variable BxHxWxC
        num_output_channels: int
        kernel_size: int
        scope: string
        stride: a list of 2 ints
        padding: 'SAME' or 'VALID'
        use_xavier: bool, use xavier_initializer if true
        stddev: float, stddev for truncated_normal init
        weight_decay: float
        activation_fn: function
        bn: bool, whether to use batch norm
        bn_decay: float or float tensor variable in [0,1]
        is_training: bool Tensor variable

    Returns:
        Variable tensor
    """
    with tf.variable_scope(scope, reuse=reuse):
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.conv1d(inputs, num_output_channels, kernel_size, stride, padding,
                                   kernel_initializer=initializer,
                                   kernel_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   bias_regularizer=tf.contrib.layers.l2_regularizer(
                                       weight_decay),
                                   use_bias=use_bias, reuse=None)
        assert not (bn and ibn)
        if bn:
            outputs = tf.layers.batch_normalization(
                outputs, momentum=bn_decay, training=is_training, renorm=False, fused=True)
            # outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
        if ibn:
            outputs = instance_norm(outputs, is_training)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

def conv2d(inputs,
           num_output_channels,
           kernel_size,
           scope=None,
           stride=[1, 1],
           padding='SAME',
           use_xavier=True,
           stddev=1e-3,
           weight_decay=0.00001,
           activation_fn=tf.nn.relu,
           bn=False,
           ibn = False,
           bn_decay=None,
           use_bias = True,
           is_training=None,
           reuse=tf.AUTO_REUSE):
  """ 2D convolution with non-linear operation.

  Args:
    inputs: 4-D tensor variable BxHxWxC
    num_output_channels: int
    kernel_size: a list of 2 ints
    scope: string
    stride: a list of 2 ints
    padding: 'SAME' or 'VALID'
    use_xavier: bool, use xavier_initializer if true
    stddev: float, stddev for truncated_normal init
    weight_decay: float
    activation_fn: function
    bn: bool, whether to use batch norm
    bn_decay: float or float tensor variable in [0,1]
    is_training: bool Tensor variable

  Returns:
    Variable tensor
  """
  with tf.variable_scope(scope,reuse=reuse) as sc:
      if use_xavier:
          initializer = tf.contrib.layers.xavier_initializer()
      else:
          initializer = tf.truncated_normal_initializer(stddev=stddev)

      outputs = tf.layers.conv2d(inputs,num_output_channels,kernel_size,stride,padding,
                                 kernel_initializer=initializer,
                                 kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                 use_bias=use_bias,reuse=None)
      assert not (bn and ibn)
      if bn:
          outputs = tf.layers.batch_normalization(outputs,momentum=bn_decay,training=is_training,renorm=False,fused=True)
          #outputs = tf.contrib.layers.batch_norm(outputs,is_training=is_training)
      if ibn:
          outputs = instance_norm(outputs,is_training)


      if activation_fn is not None:
        outputs = activation_fn(outputs)

      return outputs


def fully_connected(inputs,
                    num_outputs,
                    scope,
                    use_xavier=True,
                    stddev=1e-3,
                    weight_decay=0.00001,
                    activation_fn=tf.nn.relu,
                    bn=False,
                    bn_decay=None,
                    use_bias = True,
                    is_training=None):
    """ Fully connected layer with non-linear operation.

    Args:
      inputs: 2-D tensor BxN
      num_outputs: int

    Returns:
      Variable tensor of size B x num_outputs.
    """

    with tf.variable_scope(scope) as sc:
        if use_xavier:
            initializer = tf.contrib.layers.xavier_initializer()
        else:
            initializer = tf.truncated_normal_initializer(stddev=stddev)

        outputs = tf.layers.dense(inputs,num_outputs,
                                  use_bias=use_bias,kernel_initializer=initializer,
                                  kernel_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  bias_regularizer=tf.contrib.layers.l2_regularizer(weight_decay),
                                  reuse=None)

        if bn:
            outputs = tf.layers.batch_normalization(outputs, momentum=bn_decay, training=is_training, renorm=False)

        if activation_fn is not None:
            outputs = activation_fn(outputs)

        return outputs

from tf_ops.grouping.tf_grouping import knn_point_2
def get_edge_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, 1, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)
    point_cloud_central = tf.expand_dims(point_cloud, axis=-2)

    point_cloud_central = tf.tile(point_cloud_central, [1, 1, k, 1])

    edge_feature = tf.concat(
        [point_cloud_central, point_cloud_neighbors - point_cloud_central], axis=-1)
    return edge_feature, idx

def get_KNN_feature(point_cloud, k=16, idx=None):
    """Construct edge feature for each point
    Args:
        point_cloud: (batch_size, num_points, num_dims)
        nn_idx: (batch_size, num_points, k, 2)
        k: int
    Returns:
        edge features: (batch_size, num_points, k, num_dims)
    """
    if idx is None:
        _, idx = knn_point_2(k+1, point_cloud, point_cloud, unique=True, sort=True)
        idx = idx[:, :, 1:, :]

    # [N, P, K, Dim]
    point_cloud_neighbors = tf.gather_nd(point_cloud, idx)

    return point_cloud_neighbors, idx

def dense_conv(feature, n=3,growth_rate=64, k=16, scope='dense_conv',**kwargs):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        y, idx = get_edge_feature(feature, k=k, idx=None)  # [B N K 2*C]
        for i in range(n):
            if i == 0:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    tf.tile(tf.expand_dims(feature, axis=2), [1, 1, k, 1])], axis=-1)
            elif i == n-1:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, activation_fn=None, **kwargs),
                    y], axis=-1)
            else:
                y = tf.concat([
                    conv2d(y, growth_rate, [1, 1], padding='VALID', scope='l%d' % i, **kwargs),
                    y], axis=-1)
        y = tf.reduce_max(y, axis=-2)
        return y,idx

def normalize_point_cloud(pc):
    """
    pc [N, P, 3]
    """
    centroid = tf.reduce_mean(pc, axis=1, keep_dims=True)
    pc = pc - centroid
    furthest_distance = tf.reduce_max(
        tf.sqrt(tf.reduce_sum(pc ** 2, axis=-1, keep_dims=True)), axis=1, keep_dims=True)
    pc = pc / furthest_distance
    return pc, centroid, furthest_distance

def up_sample(x, scale_factor=2):
    _, h, w, _ = x.get_shape().as_list()
    new_size = [h * scale_factor, w * scale_factor]
    return tf.image.resize_nearest_neighbor(x, size=new_size)

def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


def flatten(input):
    return tf.reshape(input, [-1, np.prod(input.get_shape().as_list()[1:])])

def hw_flatten(x):
    return tf.reshape(x, shape=[x.shape[0], -1, x.shape[-1]])

def safe_log(x, eps=1e-12):
  return tf.log(x + eps)


def tf_covariance(data):
    ## x: [batch_size, num_point, k, 3]
    batch_size = data.get_shape()[0].value
    num_point = data.get_shape()[1].value

    mean_data = tf.reduce_mean(data, axis=2, keep_dims=True)  # (batch_size, num_point, 1, 3)
    mx = tf.matmul(tf.transpose(mean_data, perm=[0, 1, 3, 2]), mean_data)  # (batch_size, num_point, 3, 3)
    vx = tf.matmul(tf.transpose(data, perm=[0, 1, 3, 2]), data) / tf.cast(tf.shape(data)[0], tf.float32)  # (batch_size, num_point, 3, 3)
    data_cov = tf.reshape(vx - mx, shape=[batch_size, num_point, -1])

    return data_cov



def add_scalar_summary(name, value,collection='train_summary'):
    tf.summary.scalar(name, value, collections=[collection])
def add_hist_summary(name, value,collection='train_summary'):
    tf.summary.histogram(name, value, collections=[collection])

def add_train_scalar_summary(name, value):
    tf.summary.scalar(name, value, collections=['train_summary'])

def add_train_hist_summary(name, value):
    tf.summary.histogram(name, value, collections=['train_summary'])

def add_train_image_summary(name, value):
    tf.summary.image(name, value, collections=['train_summary'])


def add_valid_summary(name, value):
    avg, update = tf.metrics.mean(value)
    tf.summary.scalar(name, avg, collections=['valid_summary'])
    return update
