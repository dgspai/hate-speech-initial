import logging
import time

import pandas as pd
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from feature_extraction.lstm_featurizer import LstmFeaturizer
from feature_extraction.vocab_featurizer import VocabFeaturizer
from text_preprocessing.simple_text_preprocessor import SimpleTextPreprocessor

# Nenhum dataset tem exatamente 5666 instancias...
file_path = 'data/raw/Portuguese-Hate-Speech-Dataset-master/2019-05-28_portuguese_hate_speech_hierarchical_classification.csv'
x_col = 'text'
y_col = 'Hate.speech'
target_names = ['No hate speech', 'Hate speech']

embedding_path = 'data/models/glove/glove_s300_sample.txt'

logging.basicConfig(level=logging.DEBUG, format='%(levelname)s : %(name)s : %(asctime)s : %(message)s')


def main():
    logging.info(f"Loading data from {file_path}")
    df = pd.read_csv(file_path)
    x = df[x_col]
    y = df[y_col]
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=567, random_state=42)
    logging.info(f"Train size {len(X_train)}")
    logging.info(f"Test size {len(X_test)}")

    logging.info("Building pipeline")
    pipeline = Pipeline([
        ('text_preprocessor', SimpleTextPreprocessor()),
        ('vocab_featurizer', VocabFeaturizer(embedding_model_path=embedding_path)),
        ('lstm_featurizer', LstmFeaturizer(embedding_model_path=embedding_path)),
        # Artigo usa grid search com 9 combinações, mas pode levar muito tempo pra checar todas
        ('classifier', XGBClassifier(eta=1, gamma=1)),
    ])
    logging.info(f"Pipeline created: {pipeline}")

    logging.info("Fitting data...")
    start_time = time.time()
    pipeline.fit(X_train, y_train)
    logging.info(f"Fitting done in {time.time() - start_time} seconds.")

    logging.info("Predicting data...")
    start_time = time.time()
    y_pred = pipeline.predict(X_test)
    logging.info(f"Predicting done in {time.time() - start_time} seconds.")

    report = classification_report(y_test, y_pred, target_names=target_names)

    print(f"\nReport:\n{report}")

    for avg in ["micro", "macro"]:
        print(f"F1 {avg}: {f1_score(y_test, y_pred, average=avg)}")

    logging.info("Done!")


if __name__ == "__main__":
    main()
