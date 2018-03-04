# encoding: utf-8

'''

@author: Condy

@file: extract_text

@time: 2018/3/4 上午10:44

'''
import json


def extract_text(filename,k):
    with open(filename, 'r') as f:
        result = []
        for line in f.readlines():
            line_f = {}
            line_list = json.loads(line.strip())
            tag_feature = line_list['Hypernym']['Tag_features']
            structure_feature = line_list['Hypernym']['Structure_feature']
            frequency = line_list['Frequency']
            line_f['tag'] = tag_feature
            line_f['structure'] = structure_feature
            line_f['frequency'] = frequency
            result.append(line_f)
        result=sorted(result,key=lambda s:s['frequency'])
        negtive=result[:k]
        active=result[k:]
        x=[]
        y=[]
        for l in negtive:
            tmp = l['tag']
            tmp.extend(l['structure'])
            x.append(tmp)
            y.append(0)
        for l in active:
            tmp = l['tag']
            tmp.extend(l['structure'])
            x.append(tmp)
            y.append(1)
        return x,y


            # print(line_list['Hyponym']['Term'],tag_feature)

            # if __name__ == '__main__':
            #     extract_text('mini_data2.txt')
