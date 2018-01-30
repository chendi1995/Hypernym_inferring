# encoding: utf-8

'''

@author: Condy

@file: train_model

@time: 2018/1/21 下午3:45

'''
import tensorflow as tf
from data_flow import *
from mymodel import CustomModel
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from option import getarg
shuffle = True

if __name__ == '__main__':
    ops = getarg()
    batchsize=ops.batchsize
    data, hypon_Maxlen, hyper_Maxlen = read_txt(ops.data_file, 'stopword.txt')
    data = data_padding(data, hypon_Maxlen, hyper_Maxlen)
    train_data, test_data = train_test_split(data, test_size=0.3)
    data_size = len(train_data)
    print("data_size:%d" % data_size)
    if data_size % batchsize:
        num_batches_per_epoch = int(data_size / batchsize) + 1
    else:
        num_batches_per_epoch = int(data_size / batchsize)
    model = CustomModel(ops,hypon_Maxlen, hyper_Maxlen, embedding_size=ops.embedding_size)
    model.build_model()

    init = tf.initialize_all_variables()

    with tf.device('/cpu'):
        # gpu use config

        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(ops.epochs):
                if shuffle:
                    random.shuffle(train_data)
                for batch_num in range(num_batches_per_epoch):
                    start_index = batchsize * batch_num
                    end_index = (batch_num + 1) * batchsize
                    batch_data = train_data[start_index:end_index]
                    X, Y, D_Feature, batch_label = get_input(batch_data, len(batch_data))
                    _, loss, accuracy = sess.run([model.optimizer, model.cross_entropy, model.accuracy],
                                                 feed_dict={model.X: X, model.Y: Y, model.D: D_Feature,
                                                            model.label: batch_label})

                    # print('loss:%f   accuracy:%f' % (loss, accuracy))

                # ------------test-------------------#
                X, Y, D_Feature, batch_label = get_input(test_data, len(test_data))
                accuracy, logits = sess.run([model.accuracy, model.logits],
                                            feed_dict={model.X: X, model.Y: Y, model.D: D_Feature,
                                                       model.label: batch_label})
                print("test_acc:%f" % accuracy)
                logits = np.argmax(logits, axis=1)
                label = np.argmax(batch_label, axis=1)
                print('precision: %f    recall:%f   f1:%f' % (
                precision_score(logits, label), recall_score(logits, label), f1_score(logits, label)))
