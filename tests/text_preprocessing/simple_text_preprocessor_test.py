import unittest

from text_preprocessing.simple_text_preprocessor import SimpleTextPreprocessor


class SimpleTextPreprocessorTest(unittest.TestCase):

    def test_fit(self):
        transformer = SimpleTextPreprocessor()
        fitted_transformer = transformer.fit(None)
        self.assertEqual(fitted_transformer, transformer)

    def test_transform(self):
        transformer = SimpleTextPreprocessor()
        inputs = [
            "Ola mundo.",
            "um bom dia para você and para você"
        ]
        outputs = transformer.transform(inputs)
        self.assertEqual(len(inputs), len(outputs))
        self.assertEqual(2, len(outputs[0]))
        self.assertEqual(7, len(outputs[1]))

    def test_process_string(self):
        input_text = "Olá mundo! Comprei 1,2,3,4."
        expected_text = "olá mundo comprei 1 2 3 4"

        transformer = SimpleTextPreprocessor()
        result_text = transformer._process_string(input_text)
        self.assertEqual(expected_text, result_text)

    def test_tokenize_string(self):
        input_text = "Olá mundo and hello world!"
        expected_tokens = ["Olá", "mundo", "hello", "world", "!"]

        transformer = SimpleTextPreprocessor()
        result_tokens = transformer._simple_tokenize(input_text)

        self.assertEqual(expected_tokens, result_tokens)
