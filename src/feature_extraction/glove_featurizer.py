import logging

from gensim.models import KeyedVectors
from sklearn.base import BaseEstimator, TransformerMixin


class GloveFeaturizer(BaseEstimator, TransformerMixin):
    """
    Converts list of words into list of embeddings.
    """

    def __init__(self, model_path):
        self.model_path = model_path
        logging.info("Loading vectors from " + self.model_path)
        self.model = KeyedVectors.load_word2vec_format(self.model_path)
        logging.info("Vectors loaded!")

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = []
        for instance in X:
            vectors = []
            for word in instance:
                vectors.append(self._get_embedding(word))
            result.append(vectors)
        return result

    def _get_embedding(self, word):
        if word in self.model:
            return self.model.get_vector(word)
        return self.model.get_vector('<unk>')
