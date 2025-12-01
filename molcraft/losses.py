import warnings
import keras 
import tensorflow as tf
import numpy as np


@keras.saving.register_keras_serializable(package='molcraft')
class GaussianNegativeLogLikelihood(keras.losses.Loss):

    def __init__(
        self, 
        events: int = 1, 
        name: str = "gaussian_nll", 
        **kwargs
    ) -> None:
        super().__init__(name=name, **kwargs)
        self.events = events
    
    def call(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        mean = y_pred[..., :self.events]
        scale = y_pred[..., self.events:]
        variance = keras.ops.square(scale)
        expected_rank = len(keras.ops.shape(mean))
        current_rank = len(keras.ops.shape(y_true))
        for _ in range(expected_rank - current_rank):
            y_true = keras.ops.expand_dims(y_true, axis=-1)
        return keras.ops.mean(
            0.5 * keras.ops.log(2.0 * np.pi * variance) + 
            0.5 * keras.ops.square(y_true - mean) / variance 
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config['events'] = self.events 
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