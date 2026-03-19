import unittest
import tensorflow as tf
import keras

from molcraft import losses
from molcraft import layers


class TestLoss(unittest.TestCase):

    def setUp(self):
        self.batch_size = 8
        self.events = [1, 2, 3]
        self.input_dim = 16
        self.x_train = tf.random.normal((self.batch_size, self.input_dim))
        self.y_true = [
            tf.random.normal((self.batch_size, event)) for event in self.events 
        ]

    def test_gaussian_nll(self):
        for i in range(len(self.events)):
            head = layers.GaussianParams(events=self.events[i])
            loss_fn = losses.GaussianNLL()
            y_pred = head(self.x_train)
            self.assertEqual(y_pred.shape, (self.batch_size, self.events[i] * 2))

            loss_value = loss_fn(self.y_true[i], y_pred)
            self.assertEqual(loss_value.shape, ())
            self.assertTrue(tf.math.is_finite(loss_value))

    def test_normal_inverse_gamma_nll(self):
        for i in range(len(self.events)):
            head = layers.NormalInverseGammaParams(events=self.events[i])
            loss_fn = losses.NormalInverseGammaNLL(lam=0.2)
            y_pred = head(self.x_train)
            self.assertEqual(y_pred.shape, (self.batch_size, self.events[i] * 4))
            
            loss_value = loss_fn(self.y_true[i], y_pred)
            self.assertEqual(loss_value.shape, ())
            self.assertTrue(tf.math.is_finite(loss_value))

    def test_loss_gradient(self):
        for i in range(len(self.events)):
            head = layers.NormalInverseGammaParams(events=self.events[i])
            loss_fn = losses.NormalInverseGammaNLL(lam=0.1)
            
            with tf.GradientTape() as tape:
                y_pred = head(self.x_train)
                loss_value = loss_fn(self.y_true[i], y_pred)
                
            grads = tape.gradient(loss_value, head.trainable_variables)
            for grad in grads:
                self.assertIsNotNone(grad)
                self.assertTrue(keras.ops.all(tf.math.is_finite(grad)))

    def test_rank_alignment_resilience(self):
        for i in range(len(self.events)):
            head = layers.GaussianParams(events=self.events[i])
            g_loss = losses.GaussianNLL()
            self.assertTrue(tf.math.is_finite(g_loss(self.y_true[i].numpy().squeeze(), head(self.x_train))))

            head = layers.NormalInverseGammaParams(events=self.events[i])
            n_loss = losses.NormalInverseGammaNLL()
            self.assertTrue(tf.math.is_finite(n_loss(self.y_true[i].numpy().squeeze(), head(self.x_train))))


if __name__ == '__main__':
    unittest.main()