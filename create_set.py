# encoding: utf-8

'''

@author: Condy

@file: classify

@time: 2017/12/21 下午8:43

'''
import json
import urllib.request

baike_data = '/storage/chendi/zhishime_json/baidubaike/'
hudong_data = '/storage/chendi/zhishime_json/hudongbaike/'
wiki_data = '/storage/chendi/zhishime_json/zhwiki/'

hyperresult = {}
hyponresult = {}


def create_set(abstract_file, instance_file, ss, tag):
    lab_set = []
    abs_set = []
    lab_abs = {}
    lab_type = {}
    with open(abstract_file) as f:
        baike_abs = json.load(f)
        for l in baike_abs:
            abs_set.append(urllib.request.unquote(l['@id'][ss:]))
            lab_abs[urllib.request.unquote(l['@id'][ss:])] = l["http://zhishi.me/ontology/abstract"][0]["@value"]
    with open(instance_file) as f:
        baike_lab = json.load(f)
        for l in baike_lab:
            lab_set.append(urllib.request.unquote(l['@id'][ss:]))
            lab_type[urllib.request.unquote(l['@id'][ss:])] = [urllib.request.unquote(i[ss:]) for i in l['@type']]
        baike_term = list(set(lab_set).intersection(abs_set))
    count = 0
    for term in baike_term:
        type_list = lab_type[term]
        # inter_set = list(set(type_list).intersection(abs_set))
        inter_set = type_list
        if len(inter_set) != 0:
            hyponresult[term+'_'+tag] = (inter_set, lab_abs[term])
            for i in inter_set:
                if i in lab_abs.keys():
                    hyperresult[i+'_'+tag] = lab_abs[i]
                else:
                    hyperresult[i+'_'+tag] = None
            count = count + 1
            if count % 100 == 0:
                print(count)

        # if count==100:
        #     break


if __name__ == '__main__':
    create_set(baike_data + '3.0_baidubaike_abstracts_zh.json', baike_data + 'baidubaike_instance_types_zh.json', 37,
               'baidu')
    create_set(hudong_data + '3.0_hudongbaike_abstracts_zh.json', hudong_data + 'hudongbaike_instance_types_zh.json',
               38,
               'hudong')
    create_set(wiki_data + '2.0_zhwiki_abstracts_zh.json', wiki_data + 'zhwiki_instance_types_zh.json', 33, 'wiki')
    with open('hyperresult.json', 'w') as f:
        json.dump(hyperresult, f)

    with open('hyponresult.json', 'w') as f:
        json.dump(hyponresult, f)
