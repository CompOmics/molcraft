import keras 
import tensorflow as tf

from molcraft import models
from molcraft import tensors 
from molcraft import layers
from molcraft import ops 


@keras.saving.register_keras_serializable(package='molcraft')
class Saliency(models.GraphModel):

    def __init__(
        self, 
        model: models.GraphModel, 
        reduce_mode: str | None = 'mean',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self._orig_model = model
        self.reduce_mode = reduce_mode
        for layer in model.layers:
            if isinstance(layer, layers.GraphConv):
                break
        else:
            raise ValueError('Could not find `GraphConv` layer(s).')
        self._embedder = models.GraphModel(model.input, layer._symbolic_input)
        self._model = models.GraphModel(layer._symbolic_input, model.output)
    
    def saliency(self, graph: tensors.GraphTensor) -> tensors.GraphTensor:
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            tape.watch(graph.node['feature'])
            y_pred = self._model(graph, training=False)
            target = self.target(graph, y_pred)
        saliency = tape.gradient(target, graph.node['feature'])
        return saliency
    
    def target(self, graph: tensors.GraphTensor, y_pred: tf.Tensor) -> tf.Tensor:
        y_true = graph.context.get('label')
        if y_true is not None and len(y_true.shape) > 1: 
            return tf.gather_nd(y_pred, tf.where(y_true != 0))
        return y_pred

    def propagate(self, graph: tensors.GraphTensor) -> tensors.GraphTensor:
        graph_orig = graph
        if tensors.is_ragged(graph_orig):
            graph = graph_orig.flatten()
        graph = self._embedder(graph, training=False)
        saliency = self.saliency(graph)
        if not self.reduce_mode:
            return graph_orig.update({'node': {'saliency': saliency}})
        if self.reduce_mode.lower().startswith('sum'):
            saliency = keras.ops.sum(saliency, axis=-1)
        else:
            saliency = keras.ops.mean(saliency, axis=-1)
        return graph_orig.update({'node': {'saliency': saliency}})

    def get_config(self) -> dict:
        config = super().get_config()
        config['model'] = keras.saving.serialize_keras_object(self._orig_model)
        config['reduce_mode'] = self.reduce_mode
        return config
    
    @classmethod
    def from_config(cls, config: dict) -> 'Saliency':
        config['model'] = keras.saving.deserialize_keras_object(config['model'])
        return super().from_config(config)
    

@keras.saving.register_keras_serializable(package='molcraft')
class IntegratedSaliency(Saliency):

    def __init__(
        self, 
        model: models.GraphModel, 
        reduce_mode: str = 'mean',
        steps: int = 50, 
        **kwargs
    ) -> None:
        super().__init__(
            model=model, 
            reduce_mode=reduce_mode, 
            **kwargs
        )
        self.steps = steps 

    def saliency(self, graph: tensors.GraphTensor) -> tensors.GraphTensor:
        original_feature = graph.node['feature']
        baseline = keras.ops.zeros_like(original_feature)
        diff = original_feature - baseline
        step_size = 1.0 / tf.cast(self.steps, tf.float32)

        initial_i = tf.constant(0, dtype=tf.int32)
        initial_gradient = keras.ops.zeros_like(original_feature)
        loop_vars = (initial_i, initial_gradient)

        def cond(i, total_gradient):
            return i < self.steps

        def body(i, total_gradient):
            alpha = tf.cast(i, tf.float32) * step_size
            step_feature = baseline + alpha * diff

            step_graph = graph.update({'node': {'feature': step_feature}})

            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(step_feature)
                y_pred = self._model(step_graph, training=False)
                target = self.target(graph, y_pred)

            step_gradient = tape.gradient(target, step_feature)
            total_gradient += step_gradient

            return i + 1, total_gradient

        _, final_gradient = keras.ops.while_loop(cond, body, loop_vars)
        average_gradient = final_gradient / tf.cast(self.steps, tf.float32)
        saliency = average_gradient * diff
        return saliency
    
    def get_config(self) -> dict:
        config = super().get_config()
        config['steps'] = self.steps
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class GradCAM(Saliency):

    def saliency(self, graph: tensors.GraphTensor) -> tensors.GraphTensor:
        graph_indicator = graph.graph_indicator
        features = []
        x = graph
        with tf.GradientTape(watch_accessed_variables=False) as tape:
            for layer in self._model.layers:
                if isinstance(layer, layers.GraphNetwork):
                    x, taped_features = layer.tape_propagate(x, tape, training=False)
                    features.extend(taped_features)
                else:
                    if (
                        isinstance(layer, layers.GraphConv) and 
                        isinstance(x, tensors.GraphTensor)
                    ):
                        tape.watch(x.node['feature'])
                        features.append(x.node['feature'])
                    x = layer(x, training=False)
            y_pred = x
            target = self.target(graph, y_pred)
        gradients = tape.gradient(target, features)
        features = keras.ops.concatenate(features, axis=-1)
        gradients = keras.ops.concatenate(gradients, axis=-1)
        alpha = ops.segment_mean(gradients, graph_indicator)
        alpha = ops.gather(alpha, graph_indicator)
        saliency = keras.ops.where(gradients != 0, alpha * features, gradients)
        return saliency