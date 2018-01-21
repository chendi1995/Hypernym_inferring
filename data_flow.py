# encoding: utf-8

'''

@author: Condy

@file: data_flow

@time: 2018/1/19 下午1:03

'''
import jieba.analyse
from gensim.models import Word2Vec
import re
import numpy as np
import random


def is_chinese(uchar):
    """判断一个unicode是否是汉字"""
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


def format_str(content):
    content_str = ''
    for i in content:
        if is_chinese(i):
            content_str = content_str + i
    return content_str


def read_txt(txtname, stopword):
    model_file = "wiki.zh.text.model"
    word2vec_model = Word2Vec.load(model_file)
    hypon_Maxlen = 0
    hyper_Maxlen = 0
    feature = []
    with open(stopword, 'r') as f:
        stopword = f.readlines()
    with open(txtname, 'r') as f:
        txtline = f.readlines()
        for line in txtline:
            line_feature = {}
            hypon_out_line = []
            hyper_out_line = []
            line_list = line.strip('\n').split('\t')
            for word in jieba.lcut(format_str(line_list[1]), cut_all=False, HMM=False):
                if word + '\n' not in stopword:
                    try:
                        hypon_out_line.append(word2vec_model[word])
                    except:
                        print(word)
            for word in jieba.lcut(format_str(line_list[3]), cut_all=False, HMM=False):
                if word + '\n' not in stopword:
                    try:
                        hyper_out_line.append(word2vec_model[word])
                    except:
                        print(word)
            hypon_Maxlen = max(hypon_Maxlen, len(hypon_out_line))
            hyper_Maxlen = max(hyper_Maxlen, len(hyper_out_line))
            line_feature['word_name'] = line_list[0]
            line_feature['hypontext'] = np.array(hypon_out_line)
            line_feature['hypername'] = line_list[2]
            line_feature['hypertext'] = np.array(hyper_out_line)
            line_feature['hyper_d_feature'] = line_list[4]
            line_feature['label'] = line_list[5]
            feature.append(line_feature)
    return feature, hypon_Maxlen, hyper_Maxlen


def data_padding(data, hypon_Maxlen, hyper_Maxlen):
    for line in data:
        len_hypon = np.shape(line['hypontext'])[0]
        line['hypontext'] = np.pad(line['hypontext'], ((0, hypon_Maxlen-len_hypon), (0, 0)), 'constant')
        len_hyper = np.shape(line['hypertext'])[0]
        line['hypertext'] = np.pad(line['hypertext'], ((0, hyper_Maxlen-len_hyper), (0, 0)), 'constant')
    return data


def batch_iter(data, batchsize, num_epochs, shuffle=True):
    data_size = len(data)
    if data_size % batchsize:
        num_batches_per_epoch = int(data_size / batchsize) + 1
    else:
        num_batches_per_epoch = int(data_size / batchsize)
    for epoch in range(num_epochs):
        if shuffle:
            random.shuffle(data)
        for batch_num in range(num_batches_per_epoch):
            start_index = batchsize * batch_num
            end_index = min((batch_num + 1) * batchsize, data_size)
            yield data[start_index:end_index]


# test_code

if __name__ == '__main__':
    data, hypon_Maxlen, hyper_Maxlen = read_txt('data_example.txt', 'stopword.txt')
    data = data_padding(data, hypon_Maxlen, hyper_Maxlen)
    print(data[1]["hypertext"])


    # for i in batch_iter(data,2,5):
    #     print(i[0]['word_name'])
