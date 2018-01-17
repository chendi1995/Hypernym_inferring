# encoding: utf-8

'''

@author: Condy

@file: mymodel.py

@time: 2018/1/16 下午2:14

'''
import tensorflow as tf
from tensorflow.contrib import rnn
from six.moves import xrange


class CustomModel:
    def __init__(self, sess, X_emd, Y_emd, XMAXLEN, YMAXLEN, batch_size=100):
        self.sess = sess
        self.X_emd = X_emd
        self.Y_emd = Y_emd
        self.XMAXLEN = XMAXLEN
        self.YMAXLEN = YMAXLEN
        self.batch_size = batch_size
        self.lstm_unit = 128
        self.drop_rate=0.5
    def build_model(self):
        with tf.variable_scope("encode_x"):
            fw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            fw_lstm = rnn.DropoutWrapper(fw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
            bw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            bw_lstm = rnn.DropoutWrapper(bw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)

            x_outputs, x_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm,
                                                                 inputs=self.X_emd, dtype=tf.float32)
        with tf.variable_scope("encode_y"):
            fw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            fw_lstm = rnn.DropoutWrapper(fw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)
            bw_lstm = rnn.BasicLSTMCell(self.lstm_unit, state_is_tuple=True)
            bw_lstm = rnn.DropoutWrapper(bw_lstm, input_keep_prob=1.0, output_keep_prob=1.0, seed=None)

            y_outputs, y_state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_lstm, cell_bw=bw_lstm,
                                                                 inputs=self.Y_emd, dtype=tf.float32)

        # outputs = [batch_size,MAXLEN,2*embedding_size]

        # with tf.variable_scope("word_by_word_attention"):
        #     tmp5 = tf.transpose(y_outputs, [1, 0, 2])
        #     self.h_n = tf.gather(tmp5, int(tmp5.get_shape()[0]) - 1)
        #
        #     self.h_n_repeat = tf.expand_dims(self.h_n, 1)
        #     pattern = tf.stack([1, self.XMAXLEN, 1])
        #     self.h_n_repeat = tf.tile(self.h_n_repeat, pattern)
        #
        #     self.W_X = tf.get_variable("W_Y", shape=[self.lstm_unit, self.lstm_unit])
        #     self.W_h = tf.get_variable("W_h", shape=[self.lstm_unit, self.lstm_unit])
        #
        #     tmp1 = tf.matmul(tf.reshape(x_outputs, shape=[self.batch_size * self.XMAXLEN, self.lstm_unit]), self.W_X,
        #                      name="Wx")
        #     self.Wx = tf.reshape(tmp1, shape=[self.batch_size, self.XMAXLEN, self.lstm_unit])
        #     tmp2 = tf.matmul(tf.reshape(self.h_n_repeat, shape=[self.batch_size * self.XMAXLEN, self.lstm_unit]),
        #                      self.W_h)
        #     self.Whn = tf.reshape(tmp2, shape=[self.batch_size, self.XMAXLEN, self.lstm_unit], name="Whn")
        #     self.M = tf.tanh(tf.add(self.Wx, self.Whn), name="M")
        #     # print "M",self.M
        #
        #     # use attention
        #     self.W_att = tf.get_variable("W_att", shape=[self.lstm_unit, 1])
        #     tmp3 = tf.matmul(tf.reshape(self.M, shape=[self.batch_size * self.XMAXLEN, self.lstm_unit]), self.W_att)
        #     # need 1 here so that later can do multiplication with h x L
        #     self.att = tf.nn.softmax(
        #         tf.reshape(tmp3, shape=[self.batch_size, 1, self.XMAXLEN], name="att"))

        with tf.variable_scope("inner_attention"):
            Xn = tf.reduce_mean(x_outputs,1)
            Xn = tf.reshape(Xn,[self.batch_size,2*self.lstm_unit])
            Ubx = tf.Variable(tf.random_normal([2*self.lstm_unit,2*self.lstm_unit]),name='Ubx')
            bnx = tf.matmul(Xn,Ubx)
            mx= []
            Ucx = tf.Variable(tf.random_normal([2 * self.lstm_unit, 2 * self.lstm_unit]), name='Ucx')
            Uax = tf.Variable(tf.random_normal([2 * self.lstm_unit, 1]), name='Uax')
            x_outputs=tf.transpose(x_outputs,[1,0,2])
            for i in range(self.XMAXLEN):
                mx.append(tf.matmul(tf.nn.tanh(tf.matmul(x_outputs[i],Ucx)+bnx),Uax))
            mx=tf.reshape(mx,[self.XMAXLEN,self.batch_size])
            mx=tf.transpose(mx,[1,0])
            alpha = tf.nn.softmax(mx)
            self.sx = tf.reshape(alpha, [self.batch_size, self.XMAXLEN, 1])
            x_att = []
            for i in range(self.batch_size):
                x_att.append(tf.transpose(tf.matmul(tf.transpose(x_outputs[i]), self.s[i])))


            Yn = tf.reduce_mean(y_outputs, 1)
            Yn = tf.reshape(Yn, [self.batch_size, 2 * self.lstm_unit])
            Ub = tf.Variable(tf.random_normal([2 * self.lstm_unit, 2 * self.lstm_unit]), name='Ub')
            bn = tf.matmul(Yn, Ub)
            m = []
            Uc = tf.Variable(tf.random_normal([2 * self.lstm_unit, 2 * self.lstm_unit]), name='Uc')
            Ua = tf.Variable(tf.random_normal([2 * self.lstm_unit, 1]), name='Ua')
            y_outputs = tf.transpose(y_outputs, [1,0,2])
            for i in range(self.YMAXLEN):
                m.append(tf.matmul(tf.nn.tanh(tf.matmul(y_outputs[i], Uc) + bn), Ua))
            m = tf.reshape(m, [self.YMAXLEN, self.batch_size])
            m = tf.transpose(m, [1, 0])
            alpha = tf.nn.softmax(m)
            self.s = tf.reshape(alpha, [self.batch_size, self.YMAXLEN, 1])
            y_att = []
            for i in range(self.batch_size):
                y_att.append(tf.transpose(tf.matmul(tf.transpose(y_outputs[i]), self.s[i])))

        with tf.variable_scope("concact layer"):
            output = tf.concat([x_att,y_att],1)
            output_drop = tf.nn.dropout(output,self.drop_rate)

        with tf.variable_scope("dense"):
            logits = tf.layers.dense(inputs=output_drop, units=2, activation=None)




