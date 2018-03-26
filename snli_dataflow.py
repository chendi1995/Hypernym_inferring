# encoding: utf-8

'''

@author: Condy

@file: snli_dataflow

@time: 2018/3/26 下午2:54

'''
from gensim.models import Word2Vec
import re
import numpy as np
import random
import json
import pickle
import os


def process_data(filename,split_num,word2vec_model,typename):
    hypon_Maxlen = 0
    hyper_Maxlen = 0
    tag=0
    count=0
    with open(filename,'r') as f:
        feature=[]
        for line in f:
            line_feature = {}
            one_data = json.loads(line)
            hypon = one_data["sentence1"]
            hyper =one_data["sentence2"]
            label = one_data["gold_label"]
            hypon_out_line = []
            hyper_out_line = []
            for word in hypon.split(' '):
                try:
                    hypon_out_line.append(word2vec_model[word])
                except:
                    continue
            for word in hyper.split(' '):
                try:
                    hyper_out_line.append(word2vec_model[word])
                except:
                    continue
            hypon_Maxlen = max(hypon_Maxlen, len(hypon_out_line))
            hyper_Maxlen = max(hyper_Maxlen, len(hyper_out_line))
            line_feature['hypontext'] = np.array(hypon_out_line)
            line_feature['hypertext'] = np.array(hyper_out_line)
            line_feature['label'] = label
            if len(hypon_out_line)!=0 and len(hyper_out_line)!=0:
                feature.append(line_feature)
            count=count+1
            if count==split_num:
                pickle.dump(feature, open("/storage/chendi/dbpedia/%s_data%d.json"%(typename,tag), "wb"))
                tag=tag+1
                feature=[]
                count=0
                print('one ok!')
                continue

    length={}
    length["hypon"]=hypon_Maxlen
    length["hyper"]=hyper_Maxlen
    print(length)
    json.dump(length,open("/storage/chendi/dbpedia/length%s.json"%typename,"w"))

def data_padding(data, hypon_Maxlen, hyper_Maxlen):
    for line in data:
        len_hypon = np.shape(line['hypontext'])[0]
        line['hypontext'] = np.pad(line['hypontext'], ((0, hypon_Maxlen - len_hypon), (0, 0)), 'constant')
        len_hyper = np.shape(line['hypertext'])[0]
        line['hypertext'] = np.pad(line['hypertext'], ((0, hyper_Maxlen - len_hyper), (0, 0)), 'constant')
    return data


def get_input(batch_data, batchsize):
    batch_x = []
    batch_y = []
    batch_label = []
    for line in batch_data:
        x = line['hypontext']
        batch_x.append(x)
        y = line['hypertext']
        batch_y.append(y)
        label = line['label']
        if label=="entailment":
            label=1
        elif label=="neutral":
            label=2
        else :
            label=0
        # print('111111')
        # label = int(label)
        batch_label.append(label)
    batch_label = np.array(batch_label)
    a = np.zeros([batchsize, 3])
    a[np.arange(batchsize), batch_label] = 1
    return np.array(batch_x), np.array(batch_y), a

if __name__ == '__main__':
    model_file = "/storage/chendi/word2vec/en_hypernym_text.model"
    word2vec_model = Word2Vec.load(model_file)
    process_data(os.path.join('/storage/chendi/snli_1.0','snli_1.0_train.jsonl'),10000,word2vec_model,"train")
    process_data(os.path.join('/storage/chendi/snli_1.0','snli_1.0_test.jsonl'),10000,word2vec_model,"test")