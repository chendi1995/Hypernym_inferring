# encoding: utf-8

'''

@author: Condy

@file: test_embedding

@time: 2018/3/13 下午7:17

'''

from gensim.models import Word2Vec


model_file = "/hdd/chendi/en_hypernym_text.model"
word2vec_model = Word2Vec.load(model_file)

print(word2vec_model['Hubei'])