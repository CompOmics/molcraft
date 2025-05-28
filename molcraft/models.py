import typing
import keras
import numpy as np
import tensorflow as tf
from pathlib import Path
from keras.src.models import functional

from molcraft import layers
from molcraft import tensors
from molcraft import ops


@keras.saving.register_keras_serializable(package="molcraft")
class GraphModel(layers.GraphLayer, keras.models.Model):
    
    """A graph model.

    Currently, the `GraphModel` only supports `GraphTensor` input.

    Build a subclassed GraphModel:

    >>> import molcraft
    >>> import keras
    >>>
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer()
    >>> graph = featurizer([('N[C@@H](C)C(=O)O', 1.0), ('N[C@@H](CS)C(=O)O', 2.0)])
    >>>
    >>> @keras.saving.register_keras_serializable()
    >>> class GraphNeuralNetwork(molcraft.models.GraphModel):
    ...     def __init__(self, units, **kwargs):
    ...         super().__init__(**kwargs)
    ...         self.units = units
    ...         self.node_embedding = molcraft.layers.NodeEmbedding(self.units)
    ...         self.edge_embedding = molcraft.layers.EdgeEmbedding(self.units)
    ...         self.conv_1 = molcraft.layers.GraphTransformer(self.units)
    ...         self.conv_2 = molcraft.layers.GraphTransformer(self.units)
    ...         self.readout = molcraft.layers.Readout('mean')
    ...         self.dense = keras.layers.Dense(1)
    ...     def propagate(self, graph):
    ...         x = self.edge_embedding(self.node_embedding(graph))
    ...         x = self.conv_2(self.conv_1(x))
    ...         return self.dense(self.readout(x))
    ...     def get_config(self):
    ...         config = super().get_config()
    ...         config['units'] = self.units
    ...         return config
    >>>
    >>> model = GraphNeuralNetwork(128)
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(1e-3),
    ...     loss=keras.losses.MeanSquaredError(),
    ...     metrics=[keras.metrics.MeanAbsolutePercentageError(name='mape')]
    ... )
    >>> model.fit(graph, epochs=10)
    >>> mse, mape = model.evaluate(graph)
    >>> preds = model.predict(graph)

    Build a functional GraphModel:

    >>> import molcraft
    >>> import keras
    >>>
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer()
    >>> graph = featurizer([('N[C@@H](C)C(=O)O', 1.0), ('N[C@@H](CS)C(=O)O', 2.0)])
    >>>
    >>> inputs = molcraft.layers.Input(graph.spec)
    >>> x = molcraft.layers.NodeEmbedding(128)(inputs)
    >>> x = molcraft.layers.EdgeEmbedding(128)(x)
    >>> x = molcraft.layers.GraphTransformer(128)(x)
    >>> x = molcraft.layers.GraphTransformer(128)(x)
    >>> x = molcraft.layers.Readout('mean')(x)
    >>> outputs = keras.layers.Dense(1)(x)
    >>> model = molcraft.models.GraphModel(inputs, outputs)
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(1e-3),
    ...     loss=keras.losses.MeanSquaredError(),
    ...     metrics=[keras.metrics.MeanAbsolutePercentageError(name='mape')]
    ... )
    >>> model.fit(graph, epochs=10)
    >>> mse, mape = model.evaluate(graph)
    >>> preds = model.predict(graph)

    Build a GraphModel using `from_layers`:

    >>> import molcraft 
    >>> import keras
    >>> 
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer()
    >>> graph = featurizer([('N[C@@H](C)C(=O)O', 1.0), ('N[C@@H](CS)C(=O)O', 2.0)])
    >>> 
    >>> model = molcraft.models.GraphModel.from_layers([
    ...     molcraft.layers.Input(graph.spec),
    ...     molcraft.layers.NodeEmbedding(128),
    ...     molcraft.layers.EdgeEmbedding(128),
    ...     molcraft.layers.GraphTransformer(128),
    ...     molcraft.layers.GraphTransformer(128),
    ...     molcraft.layers.Readout('mean'),
    ...     keras.layers.Dense(1)
    ... ])
    >>> model.compile(
    ...     optimizer=keras.optimizers.Adam(1e-3),
    ...     loss=keras.losses.MeanSquaredError(),
    ...     metrics=[keras.metrics.MeanAbsolutePercentageError(name='mape')]
    ... )
    >>> model.fit(graph, epochs=10)
    >>> mse, mape = model.evaluate(graph)
    >>> preds = model.predict(graph)

    """

    def __new__(cls, *args, **kwargs):
        if _functional_init_arguments(args, kwargs) and cls == GraphModel:
            return FunctionalGraphModel(*args, **kwargs)
        return typing.cast(GraphModel, super().__new__(cls))
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.jit_compile = False 

    @classmethod
    def from_layers(cls, graph_layers: list, **kwargs):
        """Creates a graph model from a list of graph layers.

        Currently requires `molcraft.layers.Input(spec)`. 

        If `molcraft.layers.Input(spec)` is supplied, it both 
        creates and builds the layer, as a functional model.
        `molcraft.layers.Input` is a function which returns
        a nested structure of graph components based on `spec`.

        Args:
            graph_layers:
                A list of `GraphLayer` instances, except the initial element
                which is a dictionary of Keras tensors produced by 
                `molcraft.layers.Input(spec)`.
        """
        if not tensors.is_graph(graph_layers[0]):
            # TODO: Allow this. E.g.: return cls(layers=graph_layers)
            raise ValueError(
                'Graph input not found. Make sure to add `Input`.'
            )
        inputs: dict = graph_layers.pop(0)
        x = inputs
        for layer in graph_layers:
            if isinstance(layer, list):
                layer = layers.GraphNetwork(layer)
            x = layer(x)
        outputs = x
        return cls(inputs=inputs, outputs=outputs, **kwargs)
    
    def compile(
        self, 
        optimizer: keras.optimizers.Optimizer | str | None = 'rmsprop',
        loss: keras.losses.Loss | str | None = None,
        loss_weights: dict[str, float] = None,
        metrics: list[keras.metrics.Metric] = None,
        weighted_metrics: list[keras.metrics.Metric] | None = None,
        run_eagerly: bool = False,
        steps_per_execution: int = 1,
        jit_compile: str | bool = False,
        auto_scale_loss: bool = True,
        **kwargs
    ) -> None:
        """Compiles the model.

        Args:
            optimizer:
                The optimizer to be used (a `keras.optimizers.Optimizer` subclass).
            loss:
                The loss function to be used (a `keras.losses.Loss` subclass).
            metrics:
                A list of metrics to be used during training (`fit`) and evaluation
                (`evaluate`). Should be `keras.metrics.Metric` subclasses.
            kwargs:
                See `Model.compile` in Keras documentation. 
                May or may not apply here.
        """
        super().compile(
            optimizer=optimizer,
            loss=loss,
            loss_weights=loss_weights,
            metrics=metrics,
            weighted_metrics=weighted_metrics,
            run_eagerly=run_eagerly,
            steps_per_execution=steps_per_execution,
            jit_compile=jit_compile,
            auto_scale_loss=auto_scale_loss,
            **kwargs
        )

    def fit(self, x: tensors.GraphTensor | tf.data.Dataset, **kwargs):
        """Fits the model.

        Args:
            x: 
                A `GraphTensor` instance or a `tf.data.Dataset` constructed from
                a `GraphTensor` instance. In comparison to a typical Keras model,
                the label (typically denoted `y`) and the sample_weight (typically
                denoted `sample_weight`) should be encoded in the context of the 
                `GraphTensor` instance, as `label` and `weight` respectively.
            validation_data:
                A `GraphTensor` instance or a `tf.data.Dataset` constructed from
                a `GraphTensor` instance. In comparison to a typical Keras model,
                the label (typically denoted `y`) and the sample_weight (typically
                denoted `sample_weight`) should be encoded in the context of the 
                `GraphTensor` instance, as `label` and `weight` respectively.
            validaton_split:
                The fraction of training data to be used as validation data. 
                Only works if a `GraphTensor` instance is passed as `x`. 
            batch_size:
                Number of samples per batch of computation.
            epochs:
                Number of iterations over the entire dataset.
            callbacks:
                A list of callbacks to apply during training.
            kwargs:
                See `Model.fit` in Keras documentation. 
                May or may not apply here. 
        """
        batch_size = kwargs.get('batch_size', 32)
        x_val = kwargs.pop('validation_data', None)
        val_split = kwargs.pop('validation_split', None)
        if x_val is not None and isinstance(x_val, tensors.GraphTensor):
            x_val = _make_dataset(x_val, batch_size)
        if isinstance(x, tensors.GraphTensor):
            if val_split:
                val_size = int(val_split * x.num_subgraphs)
                x_val = _make_dataset(x[-val_size:], batch_size)
                x = x[:-val_size]
            x = _make_dataset(x, batch_size)
        return super().fit(x, validation_data=x_val, **kwargs)
    
    def evaluate(self, x: tensors.GraphTensor | tf.data.Dataset, **kwargs):
        """Evaluation of the model.

        Args:
            x: 
                A `GraphTensor` instance or a `tf.data.Dataset` constructed from
                a `GraphTensor` instance. In comparison to a typical Keras model,
                the label (typically denoted `y`) and the sample_weight (typically
                denoted `sample_weight`) should be encoded in the context of the 
                `GraphTensor` instance, as `label` and `weight` respectively.
            batch_size:
                Number of samples per batch of computation.
            kwargs:
                See `Model.evaluate` in Keras documentation. 
                May or may not apply here. 
        """
        batch_size = kwargs.get('batch_size', 32)
        if isinstance(x, tensors.GraphTensor):
            x = _make_dataset(x, batch_size)
        metric_results = super().evaluate(x, **kwargs)
        return tf.nest.map_structure(lambda value: float(value), metric_results)
    
    def predict(self, x: tensors.GraphTensor | tf.data.Dataset, **kwargs):
        """Makes predictions with the model.

        Args:
            x: 
                A `GraphTensor` instance or a `tf.data.Dataset` constructed from
                a `GraphTensor` instance.
            batch_size:
                Number of samples per batch of computation.
            kwargs:
                See `Model.predict` in Keras documentation. 
                May or may not apply here. 
        """
        batch_size = kwargs.get('batch_size', 32)
        if isinstance(x, tensors.GraphTensor):
            x = _make_dataset(x, batch_size)
        return super().predict(x, **kwargs)

    def get_compile_config(self) -> dict | None:
        config = super().get_compile_config()
        if config is None:
            return
        return config

    def compile_from_config(self, config: dict | None) -> None:
        if config is None:
            return
        config = keras.utils.deserialize_keras_object(config)
        self.compile(**config)
        if hasattr(self, 'optimizer') and self.built:
            self.optimizer.build(self.trainable_variables)

    def save(
        self,
        filepath: str | Path,
        *args,
        **kwargs
    ) -> None:
        """Saves an entire model.
        
        Args:
            filepath:
                A string with the path to the model file (requires `.keras` suffix)
        """
        if not self.built:
            raise ValueError('Cannot save model as it has not been built yet.')
        super().save(filepath, *args, **kwargs)

    @staticmethod
    def load(
        filepath: str | Path,
        *args,
        **kwargs
    ) -> keras.Model:
        """A `staticmethod` loading an entire model.

        Args:
            filepath:
                A string with the path to the model file (requires `.keras` suffix)
        """
        return keras.models.load_model(filepath, *args, **kwargs)
    
    def save_weights(self, filepath, *args, **kwargs):
        """Saves the weights of the model.

        Args:
            filepath:
                A string with the path to the file (requires `.weights.h5` suffix)
        """
        path = Path(filepath).parent
        path.mkdir(parents=True, exist_ok=True)
        return super().save_weights(filepath, *args, **kwargs)

    def load_weights(self, filepath, *args, **kwargs):
        """Loads the weights from file saved via `save_weights()`.
        
        Args:
            filepath:
                A string with the path to the file (requires `.weights.h5` suffix)
        """
        super().load_weights(filepath, *args, **kwargs)

    def embedding(self, layer_name: str = None) -> 'FunctionalGraphModel':
        model = self
        if not isinstance(model, FunctionalGraphModel):
            raise ValueError(
                'Currently, to extract the embedding part of the model, '
                'it needs to be a `FunctionalGraphModel`. '
            )
        inputs = model.input 
        if not layer_name:
            for layer in model.layers:
                if isinstance(layer, layers.Readout):
                    outputs = layer.output 
        else:
            layer = model.get_layer(layer_name)
            outputs = (
                layer.output if isinstance(layer, keras.layers.Layer) else None
            )
            if outputs is None:
                raise ValueError(
                    f'Could not find `{layer_name}` or '
                    f'`{layer_name} is not a `keras.layers.Layer`.'
                )
        return self.__class__(inputs, outputs, name=f'{self.name}_embedding')

    def backbone(self) -> 'FunctionalGraphModel':
        if not isinstance(self, FunctionalGraphModel):
            raise ValueError(
                'Currently, to extract the backbone part of the model, '
                'it needs to be a `FunctionalGraphModel`, with a `Readout` '
                'layer dividing the backbone and the head part of the model.'
            )
        inputs = self.input
        outputs = None
        for layer in self.layers:
            if isinstance(layer, layers.Readout):
                outputs = layer.output
        if outputs is None:
            raise ValueError(
                'Could not extract output. `Readout` layer not found.'
            )
        return self.__class__(inputs, outputs, name=f'{self.name}_head')

    def head(self) -> functional.Functional:
        if not isinstance(self, FunctionalGraphModel):
            raise ValueError(
                'Currently, to extract the head part of the model, '
                'it needs to be a `FunctionalGraphModel`, with a `Readout` '
                'layer dividing the backbone and the head part of the model.'
            )
        inputs = None
        for layer in self.layers:
            if isinstance(layer, layers.Readout):
                inputs = layer.output
        if inputs is None:
            raise ValueError(
                'Could not extract input. `Readout` layer not found.'
            )
        outputs = layer.output
        return keras.models.Model(inputs, outputs, name=f'{self.name}_head')

    def train_step(self, tensor: tensors.GraphTensor) -> dict[str, float]:
        y = tensor.context.get('label')
        sample_weight = tensor.context.get('weight')
        with tf.GradientTape() as tape:
            y_pred = self(tensor, training=True)
            loss = self.compute_loss(tensor, y, y_pred, sample_weight)
            loss = self.optimizer.scale_loss(loss)
        trainable_weights = self.trainable_weights 
        gradients = tape.gradient(loss, trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, trainable_weights))
        return self.compute_metrics(tensor, y, y_pred, sample_weight)
    
    def test_step(self, tensor: tensors.GraphTensor) -> dict[str, float]:
        y = tensor.context.get('label')
        sample_weight = tensor.context.get('weight')
        y_pred = self(tensor, training=False)
        return self.compute_metrics(tensor, y, y_pred, sample_weight)
    
    def predict_step(self, tensor: tensors.GraphTensor) -> np.ndarray:
        return self(tensor, training=False)

    def compute_loss(self, x, y, y_pred, sample_weight=None):
        y, y_pred, sample_weight = _maybe_reshape(y, y_pred, sample_weight)
        return super().compute_loss(x, y, y_pred, sample_weight)
        
    def compute_metrics(self, x, y, y_pred, sample_weight=None) -> dict[str, float]:
        loss = self.compute_loss(x, y, y_pred, sample_weight)
        metric_results = {}
        for metric in self.metrics:
            if metric.name == "loss":
                metric.update_state(loss)
                metric_results[metric.name] = metric.result()
            else:
                metric.update_state(y, y_pred, sample_weight=sample_weight)
                metric_results.update(metric.result())
        return metric_results
    

@keras.saving.register_keras_serializable(package="molcraft")
class FunctionalGraphModel(functional.Functional, GraphModel):

    @property
    def layers(self):
        return [
            layer for layer in super().layers 
            if not isinstance(layer, keras.layers.InputLayer)
        ]


def save_model(model: GraphModel, filepath: str | Path, *args, **kwargs) -> None:
    if not model.built:
        raise ValueError(
            'Model and its layers have not yet been (fully) built. '
            'Build the model before saving it: `model.build(graph_spec)` '
            'or `model(graph)`.'
        )
    keras.models.save_model(model, filepath, *args, **kwargs)

def load_model(filepath: str | Path, inputs=None, *args, **kwargs) -> GraphModel:
    return keras.models.load_model(filepath, *args, **kwargs)

def create(
    *layers: list[keras.layers.Layer]
) -> GraphModel:
    if isinstance(layers[0], list):
        layers = layers[0]
    return GraphModel.from_layers(
        list(layers)
    )

def interpret(
    model: GraphModel,
    graph_tensor: tensors.GraphTensor,
) -> tensors.GraphTensor:
    x = graph_tensor
    if tensors.is_ragged(x):
        x = x.flatten()
    graph_indicator = x.graph_indicator
    y_true = x.context.get('label')
    features = []
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        for layer in model.layers:
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
        if y_true is not None and len(y_true.shape) > 1: 
            target = tf.gather_nd(y_pred, tf.where(y_true != 0))
        else:
            target = y_pred
    gradients = tape.gradient(target, features)
    features = keras.ops.concatenate(features, axis=-1)
    gradients = keras.ops.concatenate(gradients, axis=-1)
    alpha = ops.segment_mean(gradients, graph_indicator)
    alpha = ops.gather(alpha, graph_indicator)
    maps = keras.ops.where(gradients != 0, alpha * features, gradients)
    maps = keras.ops.sum(maps, axis=-1)
    return graph_tensor.update(
        {
            'node': {
                'saliency': maps
            }
        }
    )

def saliency(
    model: GraphModel,
    graph_tensor: tensors.GraphTensor,
) -> tensors.GraphTensor:
    x = graph_tensor
    if tensors.is_ragged(x):
        x = x.flatten()
    y_true = x.context.get('label')
    with tf.GradientTape(watch_accessed_variables=False) as tape:
        tape.watch(x.node['feature'])
        y_pred = model(x, training=False)
        if y_true is not None and len(y_true.shape) > 1: 
            target = tf.gather_nd(y_pred, tf.where(y_true != 0))
        else:
            target = y_pred
    gradients = tape.gradient(target, x.node['feature'])
    gradients = keras.ops.absolute(gradients)
    return graph_tensor.update(
        {
            'node': {
                'feature_saliency': gradients
            }
        }
    ) 

def predict(
    model: GraphModel,
    x: tensors.GraphTensor | tf.data.Dataset,
    repeats: int | None = 16, 
    batch_size: int = 256,
    verbose: int = 0,
    **kwargs,
) -> tuple[tf.Tensor | np.ndarray, tf.Tensor | np.ndarray]:
    """Predict with model.

    By default performs monte-carlo predictions. Namely, it performs
    `repeats` number of predictions for each example with `training = True`,
    and subsequently computes mean and standard deviations of the predictions.

    Args:
        x: 
            A `GraphTensor` instance.
        repeats:
            Number of predictions per example.
        batch_size:
            Number of samples per batch of computation.
        kwargs:
            See `Model.predict` in Keras documentation. 
            May or may not apply here. 
    """
    if not repeats:
        return model.predict(
            x, batch_size=batch_size, verbose=verbose, **kwargs
        )
    if isinstance(x, tensors.GraphTensor):
        ds = tf.data.Dataset.from_tensor_slices(x)
        ds = ds.repeat(repeats)
        ds = ds.batch(batch_size)
    elif isinstance(x, tf.data.Dataset):
        ds = x.repeat(repeats)
    else:
        raise ValueError(
            'Input `x` needs to be a `tensors.GraphTensor` instance '
            'or a `tf.data.Dataset` instance constructed from `tensors.GraphTensor`.'
        )
    ds = ds.prefetch(-1)
    y_pred = keras.ops.concatenate([
        model(x, training=True) for x in ds])
    global_batch_size = len(y_pred) // repeats
    y_pred = np.reshape(y_pred, (repeats, global_batch_size, -1))
    y_pred_loc = keras.ops.mean(y_pred, axis=0)
    y_pred_scale = keras.ops.std(y_pred, axis=0)
    if tf.executing_eagerly():
        y_pred_loc = y_pred_loc.numpy()
        y_pred_scale = y_pred_scale.numpy()
    return (y_pred_loc, y_pred_scale)
    
def _functional_init_arguments(args, kwargs):
    return (
        (len(args) == 2)
        or (len(args) == 1 and "outputs" in kwargs)
        or ("inputs" in kwargs and "outputs" in kwargs)
    )

def _make_dataset(x: tensors.GraphTensor, batch_size: int):
    return (
        tf.data.Dataset.from_tensor_slices(x)
        .batch(batch_size)
        .prefetch(-1)
    )

def _maybe_reshape(y, y_pred, sample_weight):
    if (
        sample_weight is not None and 
        len(keras.ops.shape(sample_weight)) == 2 and 
        sample_weight.shape == y_pred.shape
    ):
        y = keras.ops.reshape(y, [-1])
        y_pred = keras.ops.reshape(y_pred, [-1])
        sample_weight = keras.ops.reshape(sample_weight, [-1])
    return y, y_pred, sample_weight