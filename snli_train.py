# encoding: utf-8

'''

@author: Condy

@file: snli_train

@time: 2018/3/26 下午4:49

'''
import tensorflow as tf
from snli_dataflow import *
from snli_model import CustomModel
from sklearn.cross_validation import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import time
import pickle
import pickle

batchsize = 128

if __name__ == '__main__':
    lenth = json.load(open("/storage/chendi/dbpedia/lengthtrain.json", "r"))
    hypon_maxlen = lenth["hypon"]
    hyper_maxlen = lenth["hyper"]
    model = CustomModel(hypon_maxlen, hyper_maxlen, embedding_size=400)
    model.build_model()
    init = tf.initialize_all_variables()
    test_data = pickle.load(open("/storage/chendi/dbpedia/test_data0.json", "rb"))
    test_data = data_padding(test_data, hypon_maxlen, hyper_maxlen)
    with tf.device('/cpu'):
        with tf.Session() as sess:
            sess.run(init)
            for epoch in range(20):
                start_time = time.time()
                for tag in range(55):
                    train_data = pickle.load(open("/storage/chendi/dbpedia/train_data%d.json" % tag, "rb"))
                    data = data_padding(train_data, hypon_maxlen, hyper_maxlen)
                    data_size = len(data)
                    if data_size % batchsize:
                        num_batches_per_epoch = int(data_size / batchsize) + 1
                    else:
                        num_batches_per_epoch = int(data_size / batchsize)

                    start_time = time.time()
                    random.shuffle(data)
                    for batch_num in range(num_batches_per_epoch):
                        start_index = batchsize * batch_num
                        end_index = (batch_num + 1) * batchsize
                        batch_data = data[start_index:end_index]
                        X, Y, batch_label = get_input(batch_data, len(batch_data))
                        _, loss, accuracy = sess.run(
                            [model.optimizer, model.cross_entropy, model.accuracy],
                            feed_dict={model.X: X, model.Y: Y,
                                       model.label: batch_label})
                        print('loss:%f   accuracy:%f' % (loss, accuracy))

                X, Y, batch_label = get_input(test_data, len(test_data))
                accuracy, logits = sess.run([model.accuracy, model.logits],
                                            feed_dict={model.X: X, model.Y: Y,
                                                       model.label: batch_label})
                print("%d epoch:" % (epoch + 1))
                print("test_acc:%f" % accuracy)
                print('epoch time %f' % (time.time() - start_time))