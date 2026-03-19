import warnings
import keras 
import tensorflow as tf
import numpy as np


@keras.saving.register_keras_serializable(package='molcraft')
class GaussianNegativeLogLikelihood(keras.losses.Loss):

    def __init__(self, name: str = "gaussian_nll", **kwargs) -> None:
        super().__init__(name=name, **kwargs)

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mean, scale = keras.ops.split(y_pred, 2, axis=-1)
        variance = keras.ops.square(scale)
        if keras.ops.ndim(y_true) < keras.ops.ndim(mean):
            y_true = keras.ops.expand_dims(y_true, axis=-1)
        nll = (
            0.5 * keras.ops.log(2.0 * np.pi * variance) +
            0.5 * keras.ops.square(y_true - mean) / variance
        )
        return keras.ops.mean(nll, axis=-1)


@keras.saving.register_keras_serializable(package='molcraft')
class NormalInverseGammaNegativeLogLikelihood(keras.losses.Loss):

    def __init__(
        self,
        lam: float = 1e-2,
        name: str = 'normal_inverse_gamma_nll',
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.lam = lam

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        gamma, v, alpha, beta = keras.ops.split(y_pred, 4, axis=-1)
        if keras.ops.ndim(y_true) < keras.ops.ndim(gamma):
            y_true = keras.ops.expand_dims(y_true, axis=-1)
        omega = 2 * beta * (1 + v)
        nll = (
            0.5 * tf.math.log(np.pi / v)
            - alpha * tf.math.log(omega)
            + (alpha + 0.5) * tf.math.log(v * tf.square(y_true - gamma) + omega)
            + tf.math.lgamma(alpha)
            - tf.math.lgamma(alpha + 0.5)
        )
        error = tf.abs(y_true - gamma)
        evidence = 2 * v + alpha
        reg = error * evidence
        return keras.ops.mean(nll + self.lam * reg, axis=-1)

    def get_config(self) -> dict:
        config = super().get_config()
        config['lam'] = self.lam
        return config 


@keras.saving.register_keras_serializable(package='molcraft')
class Contrastive(keras.losses.Loss):
    def __init__(
        self,
        margin: float = 1.0,
        name: str = 'contrastive',
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.margin = margin

    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_pred = keras.ops.convert_to_tensor(y_pred)
        y_true = keras.ops.cast(y_true, y_pred.dtype)
        return (
            y_true * keras.ops.square(y_pred) + 
            (1.0 - y_true) * keras.ops.square(
                keras.ops.maximum(self.margin - y_pred, 0.0)
            )
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config['margin'] = self.margin 
        return config 


GaussianNLL = GaussianNegativeLogLikelihood
NormalInverseGammaNLL = NormalInverseGammaNegativeLogLikelihood