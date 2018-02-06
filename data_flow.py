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
import json


def is_chinese(uchar):
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
            line_feature['hyper_d_feature'] = np.array([1, 1, 1, 1])
            line_feature['label'] = int(line_list[4].strip())
            feature.append(line_feature)
    return feature, hypon_Maxlen, hyper_Maxlen


def read_eng(txtname):
    word2vec_model = Word2Vec.load(model_file)
    hypon_Maxlen = 0
    hyper_Maxlen = 0
    feature = []
    with open(txtname, 'r') as f:
        for line in f.readlines():
            line_list = json.loads(line.strip())
            line_feature = {}
            hypon_out_line = []
            hyper_out_line = []
            for word in line_list['Hyponym']['Text_desc'].split(' '):
                hypon_out_line.append(word2vec_model[word])
            for word in line_list['Hypernym']['Text_desc'].split(' '):
                hyper_out_line.append(word2vec_model[word])
            hypon_Maxlen = max(hypon_Maxlen, len(hypon_out_line))
            hyper_Maxlen = max(hyper_Maxlen, len(hyper_out_line))
            line_feature['word_name'] = line_list['Hyponym']['Term']
            line_feature['hypontext'] = np.array(hypon_out_line)
            line_feature['hypername'] = line_list['Hyperym']['Term']
            line_feature['hypertext'] = np.array(hyper_out_line)
            line_feature['hyper_d_feature'] = np.array(line_list['Hypernym']['Structure_feature'])
            line_feature['label'] = 1
            feature.append(line_feature)
    return feature, hypon_Maxlen, hyper_Maxlen



def data_padding(data, hypon_Maxlen, hyper_Maxlen):
    for line in data:
        len_hypon = np.shape(line['hypontext'])[0]
        line['hypontext'] = np.pad(line['hypontext'], ((0, hypon_Maxlen - len_hypon), (0, 0)), 'constant')
        len_hyper = np.shape(line['hypertext'])[0]
        line['hypertext'] = np.pad(line['hypertext'], ((0, hyper_Maxlen - len_hyper), (0, 0)), 'constant')
    return data


# def batch_iter(data, batchsize, num_epochs, shuffle=True):
#     data_size = len(data)
#     if data_size % batchsize:
#         num_batches_per_epoch = int(data_size / batchsize) + 1
#     else:
#         num_batches_per_epoch = int(data_size / batchsize)
#     for epoch in range(num_epochs):
#         if shuffle:
#             random.shuffle(data)
#         for batch_num in range(num_batches_per_epoch):
#             start_index = batchsize * batch_num
#             end_index = (batch_num + 1) * batchsize
#             yield data[start_index:end_index],epoch


def get_input(batch_data, batchsize):
    batch_x = []
    batch_y = []
    batch_d_feature = []
    batch_label = []
    for line in batch_data:
        x = line['hypontext']
        batch_x.append(x)
        y = line['hypertext']
        batch_y.append(y)
        d_feature = line['hyper_d_feature']
        batch_d_feature.append(d_feature)
        label = line['label']
        # print('111111')
        # label = int(label)
        batch_label.append(label)
    batch_label = np.array(batch_label)
    a = np.zeros([batchsize, 2])
    a[np.arange(batchsize), batch_label] = 1
    return np.array(batch_x), np.array(batch_y), np.array(batch_d_feature), a


# test_code

if __name__ == '__main__':
    read_eng('mini_data2.txt')
