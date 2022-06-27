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

    def test_process_string(self):
        input_text = "Olá mundo! Comprei 1,2,3,4."
        expected_text = "olá mundo comprei 1 2 3 4"

        transformer = SimpleTextPreprocessor()
        result_text = transformer._process_string(input_text)
        self.assertEqual(expected_text, result_text)

    def test_remove_stop_words_string(self):
        input_text = "Olá mundo and hello world!"
        expected = "Olá mundo hello world !"

        transformer = SimpleTextPreprocessor()
        result = transformer._remove_stop_words(input_text)

        self.assertEqual(expected, result)
