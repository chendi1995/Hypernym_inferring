# encoding: utf-8

'''

@author: Condy

@file: store_mongo_new

@time: 2017/12/26 下午2:50

'''
from pymongo import MongoClient
import json
from pymongo.son_manipulator import AutoReference, NamespaceInjector
import urllib.request

dic_path = '/storage/chendi/THUOCL/THUOCL_'

baike_dict = ['baidu', 'hudong', 'wiki']
Type_dict = ['caijing', 'food', 'lishimingren', 'chengyu', 'it', 'medical', 'diming', 'law', 'poem']

zhuanyi_dict = {'caijing': 'Finance', 'food': 'Food', 'lishimingren': 'Historic Celebrities', 'chengyu': 'Idiom',
                'it': 'IT', 'medical': 'Medicine', 'diming': 'Place', 'law': 'Law', 'poem': 'Poetry', 'baidu': 'BDBK',
                'hudong': 'HDBK', 'wiki': 'WJBK'}

if __name__ == '__main__':
    conn = MongoClient('localhost', 27017)
    db = conn.term_data
    db.add_son_manipulator(NamespaceInjector())
    db.add_son_manipulator(AutoReference(db))
    type_num = {}
    for t in Type_dict:
        type_num[t] = 0

    Baike_Category = {
        "Baike_Category":
            [{
                "type": "Baidubaike",
                "name": "百度百科",
                "key": "BDBK"
            },
                {
                    "type": "Hudongbaike",
                    "name": "互动百科",
                    "key": "HDBK"
                },
                {
                    "type": "Wikibaike",
                    "name": "维基百科",
                    "key": "WJBK"
                }
            ]
    }
    db.Baike_Category.insert(Baike_Category)
    Categorys = []
    for type in Type_dict:
        Category = {}
        Category['Type'] = zhuanyi_dict[type]
        Category['Hyponym'] = []
        Categorys.append(Category)
    term_count = 0
    with open('hyponresult.json', 'r') as f:
        hyponresult = json.load(f)
    with open('hyperresult.json', 'r') as f:
        hyperresult = json.load(f)
    for term in hyponresult.keys():
        # if len(hyponresult[term][0]) < 5:
        #     break
        # hyper_union = hyponresult[term.split['_'][0] + '_baidu'][0] | hyponresult[term.split['_'][0] + '_hudong'][0] | \
        #               hyponresult[term.split['_'][0] + '_hudong'][0]
        term_name = term.split('_')[0]
        if term_name + '_baidu' in hyponresult.keys():
            hyper_baidu = set(hyponresult[term_name + '_baidu'][0])
        else:
            hyper_baidu = set()
        if term_name + '_hudong' in hyponresult.keys():
            hyper_hudong = set(hyponresult[term_name + '_hudong'][0])
        else:
            hyper_hudong = set()
        if term_name + '_wiki' in hyponresult.keys():
            hyper_wiki = set(hyponresult[term_name + '_wiki'][0])
        else:
            hyper_wiki = set()
        hyper_union = hyper_baidu | hyper_hudong | hyper_wiki
        if len(hyper_union) < 3:
            continue
        for Category in Categorys:
            for t in Type_dict:
                if Category['Type'] == zhuanyi_dict[t]:
                    with open(dic_path + t + '.txt', 'r') as f:
                        for line in f:
                            dic_word = line.split('\t')[0]
                            if dic_word in term:
                                hyponym = {}
                                hyponym['class'] = zhuanyi_dict[t]
                                hyponym['name'] = term.split('_')[0]
                                hyponym['text'] = hyponresult[term][1]
                                hyponym['source'] = zhuanyi_dict[term.split('_')[-1]]
                                hyponym['hypernyms'] = []
                                for m in hyper_union:
                                    if m + '_baidu' in hyperresult:
                                        hypernym = {}
                                        hypernym['name'] = m
                                        hypernym['source'] = 'BDBK'
                                        hypernym['text'] = hyperresult[m + '_baidu']
                                        hyponym['hypernyms'].append(hypernym)
                                    if m + '_hudong' in hyperresult:
                                        hypernym = {}
                                        hypernym['name'] = m
                                        hypernym['source'] = 'HDBK'
                                        hypernym['text'] = hyperresult[m + '_hudong']
                                        hyponym['hypernyms'].append(hypernym)
                                    if m + '_wiki' in hyperresult:
                                        hypernym = {}
                                        hypernym['name'] = m
                                        hypernym['source'] = 'WJBK'
                                        hypernym['text'] = hyperresult[m + '_wiki']
                                        hyponym['hypernyms'].append(hypernym)
                                db.Hyponym_hypernym.insert(hyponym)
                                Category['Hyponym'].append(hyponym)
                                type_num[t] = type_num[t] + 1
                                term_count = term_count + 1
                                if term_count % 100 == 0:
                                    print(term_count)
                                break

    Hypon_hyper = {}
    Hypon_hyper["Term_num"] = term_count
    Hypon_hyper["Category"] = Categorys
    db.Hyponym_hypernym.insert(Hypon_hyper)
    Lexicon_Items = []
    for t in Type_dict:
        Lexicon_Item = {}
        Lexicon_Item['type'] = zhuanyi_dict[t]
        Lexicon_Item['vocabulary_num'] = type_num[t]
        Lexicon_Items.append(Lexicon_Item)
    Lexicon = {}
    Lexicon['Lexicon_type'] = 9
    Lexicon['Lexicon_Items'] = Lexicon_Items
    db.Lexicon.insert(Lexicon)
