# encoding: utf-8

'''

@author: Condy

@file: LR

@time: 2018/3/4 上午11:44

'''

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from baseline.extract_text import extract_text

if __name__ == '__main__':
    x,y=extract_text('mini_data2.txt',20)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = LogisticRegression()
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)
    print(y_pred)
    print(y_test)
