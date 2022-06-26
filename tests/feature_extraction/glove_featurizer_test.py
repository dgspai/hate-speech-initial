import unittest

import numpy as np

from feature_extraction.glove_featurizer import GloveFeaturizer


class GloveFeaturizerTest(unittest.TestCase):

    def setUp(self) -> None:
        self.model_path = "resources/glove_s300_sample.txt"
        self.transformer = GloveFeaturizer(self.model_path)

    def test_constructor(self):
        self.assertEqual(self.model_path, self.transformer.model_path)

    def test_fit(self):
        fitted_transformer = self.transformer.fit(None)
        self.assertEqual(fitted_transformer, self.transformer)

    def test_transform(self):
        inputs = [
            ["olá", "mundo", "hello", "world", "!"],
            ["!", "world", "hello", "mundo", "olá"]
        ]
        outputs = self.transformer.transform(inputs)
        self.assertEqual(len(inputs), len(outputs))
        self.assertEqual(len(outputs[0]), len(outputs[1]))
        np.testing.assert_array_equal(outputs[0], outputs[1][::-1])

    def test_get_embedding_from_word(self):
        input_word = "mundo"
        expected_embedding_sum = 12.38

        result_embedding = self.transformer._get_embedding(input_word)
        self.assertAlmostEqual(expected_embedding_sum, sum(result_embedding), places=2)
