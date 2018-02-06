# encoding: utf-8

'''

@author: Condy

@file: predict.py

@time: 2018/2/1 下午3:57

'''
import pickle
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
from option import getarg


def new_model(ops, x_att, y_feature):
    with tf.variable_scope("concact_layer"):
        output = tf.concat([x_att, y_feature], 1)
        output_drop = tf.nn.dropout(output, ops.drop_rate)

    with tf.variable_scope("dense_layer"):
        logits = tf.layers.dense(inputs=output_drop, units=2, activation=None)

    return logits


def predict(hyponym, hypernym):
    ops = getarg()
    with open('present_dict.json', 'rb')as f:
        present_dict = pickle.load(f)
    x = present_dict[hyponym]
    y = present_dict[hypernym]

    x = np.expand_dims(x, axis=0)
    y = np.expand_dims(y, axis=0)

    x_att = tf.placeholder("float", [None, 2 * ops.lstm_unit])
    y_feature = tf.placeholder("float", [None, 2 * ops.lstm_unit])

    logits = new_model(ops, x_att, y_feature)

    with tf.Session() as sess:
        include = ['concat_layer', 'dense_layer']
        var_to_restore = slim.get_variables_to_restore(include=include)
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(var_to_restore)

        saver.restore(sess, ops.model_path + '4.ckpt')
        print(sess.run(logits, feed_dict={x_att: x, y_feature: y}))


if __name__ == '__main__':
    predict('苹果', '苹果属')
