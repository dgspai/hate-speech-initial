from keras import Model
from keras.layers import LSTM, Activation, Dense, Dropout
from keras.models import Sequential
from sklearn.base import BaseEstimator, TransformerMixin


class LstmFeaturizer(BaseEstimator, TransformerMixin):
    """
    Uses lstm to extract features for classification
    """

    def __init__(self):
        self.full_net_model = None
        self.intermediatenet_model = None

    def _create_neural_net(self):
        # TODO Recuper glove aqui ao inves de separado
        net = Sequential()
        net.add(Dropout(0.25))
        net.add(LSTM(50))
        net.add(Dropout(0.5))
        net.add(Dense(1))
        net.add(Activation('sigmoid'))

        net.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

        return net

    def _extract_intermediate_model(self, model):
        return Model(inputs=model.input, outputs=model.layers[-3].output)

    def fit(self, X, y=None):
        self.full_net_model = self._create_neural_net()
        self.full_net_model.fit(X, y, epochs=10, batch_size=128)
        self.intermediatenet_model = self._extract_intermediate_model(self.full_net_model)
        return self

    def transform(self, X, y=None):
        if self.full_net_model is None or self.intermediatenet_model is None:
            raise RuntimeError("You must fit before transform")
        return self.intermediatenet_model.predict(X)
