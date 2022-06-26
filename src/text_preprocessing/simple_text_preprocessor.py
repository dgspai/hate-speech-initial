import re

from gensim.parsing.preprocessing import STOPWORDS
from nltk import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class SimpleTextPreprocessor(BaseEstimator, TransformerMixin):
    """
    PreProcess text into list of tokens.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        result = []
        for instance in X:
            new_instance = self._process_string(instance)
            new_instance = self._simple_tokenize(new_instance)
            result.append(new_instance)
        return result

    @staticmethod
    def _process_string(text):
        new_text = text.lower()
        new_text = re.sub(r'[^\w\s]', ' ', new_text)
        new_text = re.sub(r'[\s]+', ' ', new_text)
        return new_text.strip()

    @staticmethod
    def _simple_tokenize(text):
        words = word_tokenize(text)
        words = [word for word in words if word not in STOPWORDS]
        return words
