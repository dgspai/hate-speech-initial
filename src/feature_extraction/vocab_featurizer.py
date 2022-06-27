import logging

from gensim.models import KeyedVectors
from keras.utils import pad_sequences
from sklearn.base import BaseEstimator, TransformerMixin


class VocabFeaturizer(BaseEstimator, TransformerMixin):
    """
    Covnerts text to padded list of indexes
    """

    def __init__(self, embedding_model_path):
        self.embedding_model_path = embedding_model_path
        logging.info("Loading vectors from " + self.embedding_model_path)
        self.embedding_model = KeyedVectors.load_word2vec_format(self.embedding_model_path)
        logging.info("Vectors loaded!")

        self.max_sequence_length = -1

    def fit(self, X, y=None):
        for instance in X:
            words = instance.split()
            if len(words) > self.max_sequence_length:
                self.max_sequence_length = len(words)
        logging.info("Max sequence length is {}".format(self.max_sequence_length))
        return self

    def transform(self, X, y=None):
        result = []

        for instance in X:
            words = instance.split()
            encodes = []
            for word in words:
                if word in self.embedding_model:
                    encodes.append(self.embedding_model.get_index(word))
                else:
                    encodes.append(self.embedding_model.get_index("<unk>"))

            if len(encodes) > self.max_sequence_length:
                encodes = encodes[:self.max_sequence_length]

            result.append(encodes)

        return pad_sequences(result, maxlen=self.max_sequence_length)
