import unittest 
import keras

from molcraft import losses


class TestLoss(unittest.TestCase):

    def test_gaussian_nll(self):
        loss = losses.GaussianNegativeLogLikelihood(events=3)
        value = loss(
            keras.ops.array([[1., 2., 3.]]),
            keras.ops.array([[2., 3., 4., 0.1, 0.2, 0.3]])
        )
        self.assertGreater(value, 0)
        self.assertEqual(len(keras.ops.shape(value)), 0)

        loss = losses.GaussianNegativeLogLikelihood(events=1)
        value = loss(
            keras.ops.array([1., 2., 3.]),
            keras.ops.array([[2., 0.1], [4., 0.2], [5., 0.3]])
        )
        self.assertGreater(value, 0)
        self.assertEqual(len(keras.ops.shape(value)), 0)