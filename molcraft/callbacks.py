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
