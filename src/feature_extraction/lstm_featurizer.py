import logging

from gensim.models import KeyedVectors
from keras import Model
from keras.layers import LSTM, Activation, Dense, Dropout, Embedding
from keras.models import Sequential
from sklearn.base import BaseEstimator, TransformerMixin


class LstmFeaturizer(BaseEstimator, TransformerMixin):
    """
    Uses lstm to extract features for classification
    """

    def __init__(self, embedding_model_path):
        self.embedding_model_path = embedding_model_path
        logging.info("Loading vectors from " + self.embedding_model_path)
        self.embedding_model = KeyedVectors.load_word2vec_format(self.embedding_model_path)
        logging.info("Vectors loaded!")

        self.max_sequence_length = -1
        self.full_net_model = None
        self.intermediate_net_model = None

    def _find_max_sequence_length(self, X):
        self.max_sequence_length = len(X[0])

    def _create_neural_net(self):
        net = Sequential()
        net.add(Embedding(len(self.embedding_model), self.embedding_model.vector_size,
                          input_length=self.max_sequence_length, trainable=False, dtype='float32'))
        net.add(Dropout(0.25))
        net.add(LSTM(50))
        net.add(Dropout(0.5))
        net.add(Dense(1))
        net.add(Activation('sigmoid'))

        net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        net.layers[0].set_weights([self.embedding_model.vectors])

        return net

    def _extract_intermediate_model(self, model):
        return Model(inputs=model.input, outputs=model.layers[-3].output)

    def fit(self, X, y=None):
        # Vocab operations
        self._find_max_sequence_length(X)

        # Fit LSTM
        self.full_net_model = self._create_neural_net()
        self.full_net_model.fit(X, y, epochs=10, batch_size=128)
        self.intermediate_net_model = self._extract_intermediate_model(self.full_net_model)
        return self

    def transform(self, X, y=None):
        if self.full_net_model is None or self.intermediate_net_model is None:
            raise RuntimeError("You must fit before transform")
        return self.intermediate_net_model.predict(X)
