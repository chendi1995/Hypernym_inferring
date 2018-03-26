# encoding: utf-8

'''

@author: Condy

@file: mymodel.py

@time: 2018/1/16 下午2:14

'''
import tensorflow as tf
from tensorflow.contrib import rnn


class CustomModel:
    def __init__(self, ops, XMAXLEN, YMAXLEN, embedding_size):
        self.XMAXLEN = XMAXLEN
        self.YMAXLEN = YMAXLEN
        self.lstm_unit = ops.lstm_unit
        self.drop_rate = ops.drop_rate
        self.embedding_size = embedding_size
        self.learning_rate = ops.lr

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
            # outputs = [batch_size,MAXLEN,2*embedding_size]



        with tf.variable_scope("inner_attention"):
            Xn = tf.reduce_mean(x_outputs, 1)
            Ubx = tf.Variable(tf.random_normal([2 * self.lstm_unit, 2 * self.lstm_unit]), name='Ubx')
            bnx = tf.matmul(Xn, Ubx)
            mx = []
            Ucx = tf.Variable(tf.random_normal([2 * self.lstm_unit, 2 * self.lstm_unit]), name='Ucx')
            Uax = tf.Variable(tf.random_normal([2 * self.lstm_unit, 1]), name='Uax')
            x_outputs = tf.transpose(x_outputs, [1, 0, 2])
            for i in range(self.XMAXLEN):
                mx.append(tf.matmul(tf.nn.tanh(tf.matmul(x_outputs[i], Ucx) + bnx), Uax))
            mx = tf.reshape(mx, [self.XMAXLEN, -1])
            mx = tf.transpose(mx, [1, 0])
            alphax = tf.nn.softmax(mx)
            # sx = tf.reshape(alphax, [self.batch_size, self.XMAXLEN, 1])
            x_outputs = tf.transpose(x_outputs, [1, 0, 2])

            sx = tf.expand_dims(alphax, -1)
            x_outputs = tf.transpose(x_outputs, [0, 2, 1])
            x_att = tf.matmul(x_outputs, sx)
            self.x_att = tf.reshape(x_att, [-1, 2 * self.lstm_unit])

            # for i in range(self.batch_size):
            #     x_att.append(tf.transpose(tf.matmul(tf.transpose(x_outputs[i]), sx[i])))
            # x_att = tf.reshape(x_att, [self.batch_size, 2 * self.lstm_unit])

            Yn = tf.reduce_mean(y_outputs, 1)
            Uby = tf.Variable(tf.random_normal([2 * self.lstm_unit, 2 * self.lstm_unit]), name='Uby')
            bny = tf.matmul(Yn, Uby)
            my = []
            Ucy = tf.Variable(tf.random_normal([2 * self.lstm_unit, 2 * self.lstm_unit]), name='Ucy')
            Uay = tf.Variable(tf.random_normal([2 * self.lstm_unit, 1]), name='Uay')
            y_outputs = tf.transpose(y_outputs, [1, 0, 2])
            for i in range(self.YMAXLEN):
                my.append(tf.matmul(tf.nn.tanh(tf.matmul(y_outputs[i], Ucy) + bny), Uay))
            my = tf.reshape(my, [self.YMAXLEN, -1])
            my = tf.transpose(my, [1, 0])
            alphay = tf.nn.softmax(my)
            y_outputs = tf.transpose(y_outputs, [1, 0, 2])

            sy = tf.expand_dims(alphay, -1)
            y_outputs = tf.transpose(y_outputs, [0, 2, 1])
            y_att = tf.matmul(y_outputs, sy)
            y_att = tf.reshape(y_att, [-1, 2 * self.lstm_unit])


        # with tf.variable_scope("concact_layer"):
        #     output = tf.concat([self.x_att, y_att], 1)
        #     output_drop = tf.nn.dropout(output, self.drop_rate)
        with tf.variable_scope("concact_layer"):
            temp1 = tf.subtract(self.x_att,y_att)
            temp2 = tf.multiply(self.x_att,y_att)
            output = tf.concat([self.x_att, y_att,temp1,temp2], 1)
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
