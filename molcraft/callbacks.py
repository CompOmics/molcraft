import keras
import warnings
import numpy as np


class TensorBoard(keras.callbacks.TensorBoard):

    def _log_weights(self, epoch):
        with self._train_writer.as_default():
            for layer in self.model.layers:
                for weight in layer.weights:
                    # Use weight.path istead of weight.name to distinguish
                    # weights of different layers.
                    histogram_weight_name = weight.path + "/histogram"
                    self.summary.histogram(
                        histogram_weight_name, weight, step=epoch
                    )
                    if self.write_images:
                        image_weight_name = weight.path + "/image"
                        self._log_weight_as_image(
                            weight, image_weight_name, epoch
                        )
            self._train_writer.flush()


class LearningRateDecay(keras.callbacks.LearningRateScheduler):

    def __init__(self, rate: float, delay: int = 0, **kwargs):

        def lr_schedule(epoch: int, lr: float):
            if epoch < delay:
                return float(lr)
            return float(lr * keras.ops.exp(-rate))
        
        super().__init__(schedule=lr_schedule, **kwargs)


class Rollback(keras.callbacks.Callback):
    """Rollback callback.

    Currently, this callback simply restores the model and (optionally) the optimizer 
    variables if current loss deviates too much from the best observed loss.

    This callback might be useful in situations where the loss tend to spike and put 
    the model in an undesired/problematic high-loss parameter space.

    Args:
        tolerance (float):
            The threshold for when the restoration is triggered. The devaiation is 
            calculated as follows: (current_loss - best_loss) / best_loss.
    """

    def __init__(
        self,
        tolerance: float = 0.5,
        rollback_optimizer: bool = True,
    ):
        super().__init__()
        self.tolerance = tolerance
        self.rollback_optimizer = rollback_optimizer

    def on_train_begin(self, logs=None):
        self._rollback_weights = self._get_model_vars()
        if self.rollback_optimizer:
            self._rollback_optimizer_vars = self._get_optimizer_vars()
        self._rollback_loss = float('inf')

    def on_epoch_end(self, epoch: int, logs: dict = None):
        current_loss = logs.get('val_loss', logs.get('loss'))
        deviation = (current_loss - self._rollback_loss) / self._rollback_loss

        if np.isnan(current_loss) or np.isinf(current_loss):
            self._rollback()
            # Rolling back model because of nan or inf loss
            return

        if deviation > self.tolerance:
            self._rollback()
            # Rolling back model because of large loss deviation.
            return

        if current_loss < self._rollback_loss:
            self._save_state(current_loss)

    def _save_state(self, current_loss: float) -> None:
        self._rollback_loss = current_loss
        self._rollback_weights = self._get_model_vars()
        if self.rollback_optimizer:
            self._rollback_optimizer_vars = self._get_optimizer_vars()

    def _rollback(self) -> None:
        self.model.set_weights(self._rollback_weights)
        if self.rollback_optimizer:
            self.model.optimizer.set_weights(self._rollback_optimizer_vars)

    def _get_optimizer_vars(self):
        return [v.numpy() for v in self.model.optimizer.variables]

    def _get_model_vars(self):
        return self.model.get_weights()
