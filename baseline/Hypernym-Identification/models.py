"""Contains classifier."""

import sklearn.svm
import numpy as np
from gensim.models import Word2Vec

class DynamicMarginModel(sklearn.svm.SVC):
    """Wrapper for sklearn.svm.SVC class."""
    def __init__(self,word2vec_model, *args, **kwargs):
        self.word2vec_model = word2vec_model
        super().__init__(*args, **kwargs)


    def fit(self, X, *args, **kwargs):
        X = self.word_to_vector(X)
        super().fit(X, *args, **kwargs)

    def predict(self, X):
        X = self.word_to_vector(X)
        return super().predict(X)

    def score(self, X, *args, **kwargs):
        X = self.word_to_vector(X)
        return super().score(X, *args, **kwargs)

    def word_to_vector(self, X):
        """Converts pair of words to concatenation of their embeddings."""
        if not isinstance(X[0][0], str):
            return X

        X_embeddings = []
        for word1, word2 in X:
            embedding1 = self.word2vec_model[word1]
            embedding2 = self.word2vec_model[word2]
            norm = [np.linalg.norm(embedding1 - embedding2, ord=1)]
            X_embeddings.append(np.concatenate([embedding1, embedding2, norm]))
        return X_embeddings
