# encoding: utf-8

'''

@author: Condy

@file: SVM-Linear

@time: 2018/3/4 下午4:32

'''

from sklearn.svm import SVC

from sklearn.cross_validation import train_test_split
from baseline.extract_text import extract_text
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


if __name__ == '__main__':

    x,y=extract_text('mini_data2.txt',20)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)
    clf = SVC(kernel='linear')
    clf.fit(x_train, y_train)
    y_pred = clf.predict(x_test)

    print('precision: %f    recall:%f   f1:%f' % (
        precision_score(y_pred, y_test), recall_score(y_pred, y_test), f1_score(y_pred, y_test)))