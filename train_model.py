# encoding: utf-8

'''

@author: Condy

@file: train_model

@time: 2018/1/21 下午3:45

'''
import tensorflow as tf
from data_flow import *
from mymodel import CustomModel

batch_size = 3
nums_epoch = 5

if __name__ == '__main__':
    data, hypon_Maxlen, hyper_Maxlen = read_txt('data_example.txt', 'stopword.txt')
    data = data_padding(data, hypon_Maxlen, hyper_Maxlen)
    model = CustomModel(hypon_Maxlen, hyper_Maxlen, embedding_size=400, batch_size=batch_size)
    model.build_model()

    init = tf.initialize_all_variables()

    with tf.device('/cpu'):
        # gpu use config

        with tf.Session() as sess:
            sess.run(init)
            for batch_data in batch_iter(data, batch_size, nums_epoch):
                X, Y, D_Feature, batch_label = get_input(batch_data, len(batch_data))
                model.batch_size = len(batch_data)
                _,loss,accuracy = sess.run([model.logits,model.cross_entropy,model.accuracy], feed_dict={model.X: X, model.Y: Y, model.D: D_Feature})
