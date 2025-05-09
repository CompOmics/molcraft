import keras 
import numpy as np


@keras.saving.register_keras_serializable(package='molgraph')
class GaussianNegativeLogLikelihood(keras.losses.Loss):

    def __init__(
        self, 
        events: int = 1, 
        name="gaussian_nll", 
        **kwargs
    ):
        super().__init__(name=name, **kwargs)
        self.events = events
    
    def call(self, y_true, y_pred):
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

    def get_config(self):
        config = super().get_config()
        config['events'] = self.events 
        return config 
    

GaussianNLL = GaussianNegativeLogLikelihood