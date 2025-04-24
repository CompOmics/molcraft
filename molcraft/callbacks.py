import keras


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
