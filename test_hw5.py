import os
import sys
import unittest
import numpy as np
import timeout_decorator

from hw5 import RTVSlo, read_json, main


class TestHW5(unittest.TestCase):

    def setUp(self):
        self.data = read_json('test_dummy_data.json.gz')

    @timeout_decorator.timeout(600)
    def test_predictions(self):
        rtv = RTVSlo()
        rtv.fit(self.data )
        predictions = rtv.predict(self.data )

        # Check if the output is a numpy array
        self.assertIsInstance(predictions, np.ndarray)

        # check if array is non-empty
        self.assertNotEqual(predictions.size, 0)

        # check if the shapes are correct
        self.assertEqual(predictions.shape[0], len(self.data))

    @timeout_decorator.timeout(600)
    def test_main(self):
        sys.argv = ['hw5.py', 'test_dummy_data.json.gz', 'test_dummy_data.json.gz']
        main()

        # check if prediction.txt exists
        self.assertTrue(os.path.exists('predictions.txt'))

        # check if dimensions of test data and predictions are the same
        predictions = np.loadtxt('predictions.txt')
        self.assertEqual(len(self.data), len(predictions))

        os.remove('predictions.txt')


if __name__ == '__main__':
    unittest.main(verbosity=2)
