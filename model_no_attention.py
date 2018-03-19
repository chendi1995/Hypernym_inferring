# encoding: utf-8

'''

@author: Condy

@file: mymodel.py

@time: 2018/1/16 下午2:14

'''
import tensorflow as tf
from tensorflow.contrib import rnn


class CustomModel:
    def __init__(self, ops, XMAXLEN, YMAXLEN, embedding_size, learning_rate=0.1):
        self.XMAXLEN = XMAXLEN
        self.YMAXLEN = YMAXLEN
        self.lstm_unit = ops.lstm_unit
        self.drop_rate = ops.drop_rate
        self.embedding_size = embedding_size
        self.learning_rate = learning_rate

    def build_model(self):
        with tf.variable_scope("input_layer"):
            self.X = tf.placeholder("float", [None, self.XMAXLEN, self.embedding_size])
            self.Y = tf.placeholder("float", [None, self.
                                   YMAXLEN, self.embedding_size])
            self.D = tf.placeholder("float", [None, 4])
            self.label = tf.placeholder("float", [None, 2])
        with tf.variable_scope("encode_x"):
            fw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            fw_lstm = rnn.DropoutWrapper(fw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
            bw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            bw_lstm = rnn.DropoutWrapper(bw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)

            x_outputs, x_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm,
                                                                 inputs=self.X, dtype=tf.float32)
        with tf.variable_scope("encode_y"):
            fw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            fw_lstm = rnn.DropoutWrapper(fw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
            bw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            bw_lstm = rnn.DropoutWrapper(bw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)

            y_outputs, y_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm,
                                                                 inputs=self.Y, dtype=tf.float32)
            x_outputs = tf.concat(x_outputs, 2)
            y_outputs = tf.concat(y_outputs, 2)


        self.x_att = x_outputs[:,-1,:]

            # for i in range(self.batch_size):
            #     x_att.append(tf.transpose(tf.matmul(tf.transpose(x_outputs[i]), sx[i])))
            # x_att = tf.reshape(x_att, [self.batch_size, 2 * self.lstm_unit])
        y_att = y_outputs[:,-1,:]

        with tf.variable_scope("concat_hyper_feature"):
            D_feature = tf.layers.dense(inputs=self.D, units=2 * self.lstm_unit, activation=None)
            self.y_feature = tf.add(D_feature, y_att)

        with tf.variable_scope("concact_layer"):
            output = tf.concat([self.x_att, self.y_feature], 1)
            output_drop = tf.nn.dropout(output, self.drop_rate)

        with tf.variable_scope("dense_layer"):
            self.logits = tf.layers.dense(inputs=output_drop, units=2, activation=None)

        with tf.variable_scope("loss"):
            self.cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))

        with tf.variable_scope("optimizer"):
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cross_entropy)
            correct_pred = tf.equal(tf.argmax(self.logits, 1), tf.argmax(self.label, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
