# encoding: utf-8

'''

@author: Condy

@file: option

@time: 2018/1/30 下午7:46

'''

import argparse


def getarg():
    parser = argparse.ArgumentParser()
    parser.add_argument('data_file')
    parser.add_argument('--embedding_size',type=int)
    parser.add_argument('--lstm_unit',type=int)
    parser.add_argument('--batchsize', type=int)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--drop_rate',type=float,default=0.5)
    parser.add_argument('--shuffle', type=bool, default=True)
    args = parser.parse_args()
    return args

