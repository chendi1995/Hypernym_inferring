"""Use BLESS dataset to train classifier and save it to disk."""

import os.path
import json
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.externals import joblib
from gensim.models import Word2Vec
from models import DynamicMarginModel

def process_data(filename,word2vec_model):


    hypernym_pairs = []
    with open(filename, 'r') as file:
        for line in file:
            try:
                line_list = json.loads(line.strip())
            except:
                continue
            hyponym_ = line_list['Hyponym']['Term']
            m = re.search("<http://dbpedia.org/resource/(.*),(.*)>",hyponym_)
            if m:
                hyponym = m.group(1)
            else:
                hyponym = re.search("<http://dbpedia.org/resource/(.*)>",hyponym_).group(1)
            hyponym = hyponym.split('_')[-1]

            hypernym_ = line_list['Hypernym']['Term']
            m = re.search("<http://dbpedia.org/resource/(.*),(.*)>", hypernym_)
            if m:
                hypernym = m.group(1)
            else:
                hypernym = re.search("<http://dbpedia.org/resource/(.*)>", hypernym_).group(1)
            hypernym= hypernym.split('_')[-1]

            try:
                embedding1 = word2vec_model[hyponym]
                embedding2 = word2vec_model[hypernym]
            except:
                continue
            hypernym_pairs.append((hyponym,hypernym))



    return hypernym_pairs


if __name__ == "__main__":
    model_file = "/hdd/chendi/en_hypernym_text.model"
    word2vec_model = Word2Vec.load(model_file)
    hypernym_pairs = process_data(os.path.join('/hdd/chendi/dbpedia', 'data.txt'),word2vec_model)
    neg_pairs = process_data(os.path.join('/hdd/chendi/dbpedia', 'neg.txt'),word2vec_model)

    print(len(hypernym_pairs))
    print(len(neg_pairs))
    X = []
    X.extend(hypernym_pairs)
    X.extend(neg_pairs)
    # X.extend(meronym_pairs)
    # X.extend(random_pairs)
    #
    y = []
    y.extend([1 for _ in range(len(hypernym_pairs))])
    y.extend([0 for _ in range(len(neg_pairs))])
    # y.extend([0 for _ in range(len(meronym_pairs))])
    # y.extend([0 for _ in range(len(random_pairs))])
    #
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, test_size=0.2)

    model = DynamicMarginModel(word2vec_model,C=8, class_weight='balanced')

    model.fit(X_train, y_train)
    print('train done!')
    print('Train score: {}'.format(model.score(X_train, y_train)))
    print('Test score: {}'.format(model.score(X_test, y_test)))
    print(classification_report(y_test, model.predict(X_test)))

    model.fit(X, y)
    joblib.dump(model, os.path.join('data', 'trained_model.pkl'))
