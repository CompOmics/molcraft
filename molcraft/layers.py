import keras 
import tensorflow as tf
import warnings
import functools
from keras.src.models import functional

from molcraft import tensors
from molcraft import ops 


@keras.saving.register_keras_serializable(package='molcraft')
class GraphLayer(keras.layers.Layer):
    """Base graph layer.

    Subclasses must implement a forward pass via **propagate(graph)**. 
    
    Subclasses may create dense layers and weights in **build(graph_spec)**.

    Note: `GraphLayer` currently only supports `GraphTensor` input.

    The list of arguments below are only relevant if the derived layer 
    invokes 'get_dense_kwargs`, `get_dense`  or `get_einsum_dense`. 

    Arguments:
        use_bias (bool):
            Whether bias should be used in dense layers. Default to `True`.
        kernel_initializer (keras.initializers.Initializer, str):
            Initializer for the kernel weight matrix of the dense layers.
            Default to `glorot_uniform`.
        bias_initializer (keras.initializers.Initializer, str):
            Initializer for the bias weight vector of the dense layers.
            Default to `zeros`.
        kernel_regularizer (keras.regularizers.Regularizer, None):
            Regularizer function applied to the kernel weight matrix.
            Default to `None`.
        bias_regularizer (keras.regularizers.Regularizer, None):
            Regularizer function applied to the bias weight vector.
            Default to `None`.
        activity_regularizer (keras.regularizers.Regularizer, None):
            Regularizer function applied to the output of the dense layers.
            Default to `None`.
        kernel_constraint (keras.constraints.Constraint, None):
            Constraint function applied to the kernel weight matrix.
            Default to `None`.
        bias_constraint (keras.constraints.Constraint, None):
            Constraint function applied to the bias weight vector.
            Default to `None`.
    """

    def __init__(
        self,
        use_bias: bool = True,
        kernel_initializer: keras.initializers.Initializer | str = "glorot_uniform",
        bias_initializer: keras.initializers.Initializer | str = "zeros",
        kernel_regularizer: keras.regularizers.Regularizer | None = None,
        bias_regularizer: keras.regularizers.Regularizer | None = None,
        activity_regularizer: keras.regularizers.Regularizer | None = None,
        kernel_constraint: keras.constraints.Constraint | None = None,
        bias_constraint: keras.constraints.Constraint | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._use_bias = use_bias
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._activity_regularizer = keras.regularizers.get(activity_regularizer)
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)
        self._custom_build_config = {}
        self.built = False

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        subclass_build = cls.build

        @functools.wraps(subclass_build)
        def build_wrapper(self: GraphLayer, spec: tensors.GraphTensor.Spec | None):
            GraphLayer.build(self, spec)
            subclass_build(self, spec)
            if not self.built and isinstance(self, keras.Model):
                symbolic_inputs = Input(spec)
                self.built = True
                self(symbolic_inputs)

        cls.build = build_wrapper

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Forward pass.

        Must be implemented by subclass.

        Arguments:
            tensor:
                A `GraphTensor` instance.
        """
        raise NotImplementedError(
            'The forward pass of the layer is not implemented. '
            'Please implement `propagate`.'
        )

    def build(self, tensor_spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.

        May use built-in methods such as `get_weight`, `get_dense` and `get_einsum_dense`.

        Optionally implemented by subclass.

        Arguments:
            tensor_spec:
                A `GraphTensor.Spec` instance corresponding to the `GraphTensor` 
                passed to `propagate`.
        """ 
        if isinstance(tensor_spec, tensors.GraphTensor.Spec):
            self._custom_build_config['spec'] = _serialize_spec(tensor_spec)

    def call(
        self, 
        graph: dict[str, dict[str, tf.Tensor]]
    ) -> dict[str, dict[str, tf.Tensor]]:
        graph_tensor = tensors.from_dict(graph)
        outputs = self.propagate(graph_tensor)
        if isinstance(outputs, tensors.GraphTensor):
            return tensors.to_dict(outputs)
        return outputs

    def __call__(
        self, 
        graph: dict[str, dict[str, tf.Tensor]] | tensors.GraphTensor, 
        **kwargs
    ) -> tf.Tensor | dict[str, dict[str, tf.Tensor]] | tensors.GraphTensor:
        if not self.built:
            spec = _spec_from_inputs(graph)
            self.build(spec)

        is_graph_tensor = isinstance(graph, tensors.GraphTensor)
        if is_graph_tensor:
            graph = tensors.to_dict(graph)
        else:
            graph = {field: dict(data) for (field, data) in graph.items()}

        if isinstance(self, functional.Functional):
            # As a functional model is strict for what input can 
            # be passed to it, we need to temporarily pop some of the 
            # input and add it afterwards.
            label = graph['context'].pop('label', None)
            weight = graph['context'].pop('weight', None)
            tf.nest.assert_same_structure(self.input, graph)

        outputs = super().__call__(graph, **kwargs)

        if not tensors.is_graph(outputs):
            return outputs
        
        graph = outputs
        if isinstance(self, functional.Functional):
            if label is not None:
                graph['context']['label'] = label 
            if weight is not None:
                graph['context']['weight'] = weight

        if is_graph_tensor:
            return tensors.from_dict(graph)

        return graph

    def get_build_config(self) -> dict:
        if self._custom_build_config:
            return self._custom_build_config
        return super().get_build_config()
    
    def build_from_config(self, config: dict) -> None:
        serialized_spec = config.get('spec')
        if serialized_spec is not None:
            spec = _deserialize_spec(serialized_spec)
            self.build(spec)
        else:
            super().build_from_config(config)

    def get_weight(
        self,
        shape: tf.TensorShape,
        **kwargs,
    ) -> tf.Variable:
        common_kwargs = self.get_dense_kwargs()
        weight_kwargs = {
            'initializer': common_kwargs['kernel_initializer'],
            'regularizer': common_kwargs['kernel_regularizer'],
            'constraint': common_kwargs['kernel_constraint']
        }
        weight_kwargs.update(kwargs)
        return self.add_weight(shape=shape, **weight_kwargs)
    
    def get_dense(
        self, 
        units: int, 
        **kwargs
    ) -> keras.layers.Dense:
        common_kwargs = self.get_dense_kwargs()
        common_kwargs.update(kwargs)
        return keras.layers.Dense(units, **common_kwargs)
    
    def get_einsum_dense(
        self, 
        equation: str, 
        output_shape: tf.TensorShape, 
        **kwargs
    ) -> keras.layers.EinsumDense:
        common_kwargs = self.get_dense_kwargs()
        common_kwargs.update(kwargs)
        use_bias = common_kwargs.pop('use_bias', False)
        if use_bias and not 'bias_axes' in common_kwargs:
            common_kwargs['bias_axes'] = equation.split('->')[-1][1:] or None
        return keras.layers.EinsumDense(equation, output_shape, **common_kwargs)
    
    def get_dense_kwargs(self) -> dict:
        common_kwargs = dict(
            use_bias=self._use_bias,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        kernel_initializer = self._kernel_initializer.__class__.from_config(
            self._kernel_initializer.get_config()
        )
        bias_initializer = self._bias_initializer.__class__.from_config(
            self._bias_initializer.get_config()
        )
        common_kwargs["kernel_initializer"] = kernel_initializer
        common_kwargs["bias_initializer"] = bias_initializer
        return common_kwargs
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "use_bias": self._use_bias,
            "kernel_initializer": 
                keras.initializers.serialize(self._kernel_initializer),
            "bias_initializer": 
                keras.initializers.serialize(self._bias_initializer),
            "kernel_regularizer": 
                keras.regularizers.serialize(self._kernel_regularizer),
            "bias_regularizer": 
                keras.regularizers.serialize(self._bias_regularizer),
            "activity_regularizer":
                keras.regularizers.serialize(self._activity_regularizer),
            "kernel_constraint": 
                keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint": 
                keras.constraints.serialize(self._bias_constraint),
        })
        return config


@keras.saving.register_keras_serializable(package='molcraft')
class GraphConv(GraphLayer):

    """Base graph neural network layer.

    This layer implements the three basic steps of a graph neural network layer, each of which 
    can (optionally) be overridden by the `GraphConv` subclass:

    1. **message(graph)**, which computes the *messages* to be passed to target nodes;
    2. **aggregate(graph)**, which *aggregates* messages to target nodes;
    3. **update(graph)**, which further *updates* (target) nodes.

    Note: for skip connection to work, the `GraphConv` subclass requires final node feature 
    output dimension to be equal to `units`. 

    Arguments:
        units (int):
            Dimensionality of the output space.
        activation (keras.layers.Activation, str, None):
            Activation function to be accessed via `self.activation`, and used for the 
            `message()` and `update()` methods, if not overriden. Default to `relu`.
        use_bias (bool):
            Whether bias should be used in the dense layers. Default to `True`.
        normalize (bool, str):
            Whether normalization should be applied to the final output. Default to `False`.
        skip_connect (bool):
            Whether node feature input should be added to the node feature output. Default to `True`.
        kernel_initializer (keras.initializers.Initializer, str):
            Initializer for the kernel weight matrix of the dense layers.
            Default to `glorot_uniform`.
        bias_initializer (keras.initializers.Initializer, str):
            Initializer for the bias weight vector of the dense layers.
            Default to `zeros`.
        kernel_regularizer (keras.regularizers.Regularizer, None):
            Regularizer function applied to the kernel weight matrix.
            Default to `None`.
        bias_regularizer (keras.regularizers.Regularizer, None):
            Regularizer function applied to the bias weight vector.
            Default to `None`.
        activity_regularizer (keras.regularizers.Regularizer, None):
            Regularizer function applied to the output of the dense layers.
            Default to `None`.
        kernel_constraint (keras.constraints.Constraint, None):
            Constraint function applied to the kernel weight matrix.
            Default to `None`.
        bias_constraint (keras.constraints.Constraint, None):
            Constraint function applied to the bias weight vector.
            Default to `None`.
    """
        
    def __init__(
        self, 
        units: int = None, 
        activation: str | keras.layers.Activation | None = 'relu',
        use_bias: bool = True,
        normalize: bool = False,
        skip_connect: bool = True,
        **kwargs
    ) -> None:
        super().__init__(use_bias=use_bias, **kwargs)
        self._units = units
        self._normalize = normalize
        self._skip_connect = skip_connect
        self._activation = keras.activations.get(activation)

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        subclass_build = cls.build

        @functools.wraps(subclass_build)
        def build_wrapper(self, spec):
            GraphConv.build(self, spec)
            return subclass_build(self, spec)

        cls.build = build_wrapper

    @property 
    def units(self):
        return self._units 
    
    @property
    def activation(self):
        return self._activation
    
    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        if not self.units:
            raise ValueError(
                f'`self.units` needs to be a positive integer. Found: {self.units}.'
            )
        node_feature_dim = spec.node['feature'].shape[-1]
        self._project_residual = (
            self._skip_connect and (node_feature_dim != self.units)
        )
        if self._project_residual:
            warnings.warn(
                '`skip_connect` is set to `True`, but found incompatible dim '
                'between input (node feature dim) and output (`self.units`). '
                'Automatically applying a projection layer to residual to '
                'match input and output. ',
                stacklevel=2,
            )
            self._residual_dense = self.get_dense(
                self.units, name='residual_dense'
            )

        self.has_edge_feature = 'feature' in spec.edge
        self.has_node_coordinate = 'coordinate' in spec.node

        has_overridden_message = self.__class__.message != GraphConv.message 
        if not has_overridden_message:
            self._message_intermediate_dense = self.get_dense(self.units)
            self._message_intermediate_activation = self.activation
            self._message_final_dense = self.get_dense(self.units)

        has_overridden_aggregate = self.__class__.message != GraphConv.aggregate
        if not has_overridden_aggregate:
            pass

        has_overridden_update = self.__class__.update != GraphConv.update 
        if not has_overridden_update:
            self._update_intermediate_dense = self.get_dense(self.units)
            self._update_intermediate_activation = self.activation
            self._update_final_dense = self.get_dense(self.units)

        if not self._normalize:
            self._normalization = keras.layers.Identity()
        elif str(self._normalize).lower().startswith('layer'):
            self._normalization = keras.layers.LayerNormalization()
        else:
            self._normalization = keras.layers.BatchNormalization()

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Forward pass.

        Invokes `message(graph)`, `aggregate(graph)` and `update(graph)` in sequence.

        Arguments:
            tensor:
                A `GraphTensor` instance.
        """
        if self._skip_connect:
            residual = tensor.node['feature']
            if self._project_residual:
                residual = self._residual_dense(residual)

        message = self.message(tensor)
        add_message = not isinstance(message, tensors.GraphTensor)
        if add_message:
            message = tensor.update({'edge': {'message': message}})
        elif not 'message' in message.edge:
            raise ValueError('Could not find `message` in `edge` output.')
        
        aggregate = self.aggregate(message)
        add_aggregate = not isinstance(aggregate, tensors.GraphTensor)
        if add_aggregate:
            aggregate = tensor.update({'node': {'aggregate': aggregate}})
        elif not 'aggregate' in aggregate.node:
            raise ValueError('Could not find `aggregate` in `node` output.')
            
        update = self.update(aggregate)
        if not isinstance(update, tensors.GraphTensor):
            update = tensor.update({'node': {'feature': update}})
        elif not 'feature' in update.node:
            raise ValueError('Could not find `feature` in `node` output.')
        
        if update.node['feature'].shape[-1] != self.units:
            raise ValueError('Updated node `feature` is not equal to `self.units`.')

        if add_message and add_aggregate:
            update = update.update({'node': {'aggregate': None}, 'edge': {'message': None}})
        elif add_message:
            update = update.update({'edge': {'message': None}})
        elif add_aggregate:
            update = update.update({'node': {'aggregate': None}})

        if not self._skip_connect and not self._normalize:
            return update
        
        feature = update.node['feature']

        if self._skip_connect:
            feature += residual 

        feature = self._normalization(feature)

        return update.update({'node': {'feature': feature}})

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Compute messages.

        This method may be overridden by subclass.

        Arguments:
            tensor:
                The inputted `GraphTensor` instance.
        """
        message = keras.ops.concatenate(
            [
                tensor.gather('feature', 'source'),
                tensor.gather('feature', 'target'),
            ],
            axis=-1
        )
        if self.has_edge_feature:
            message = keras.ops.concatenate(
                [
                    message, 
                    tensor.edge['feature']
                ], 
                axis=-1
            )
        if self.has_node_coordinate:
            euclidean_distance = ops.euclidean_distance(
                tensor.gather('coordinate', 'target'),
                tensor.gather('coordinate', 'source'),
                axis=-1
            )
            message = keras.ops.concatenate(
                [
                    message,
                    euclidean_distance
                ],
                axis=-1
            )
        message = self._message_intermediate_dense(message)
        message = self._message_intermediate_activation(message)
        message = self._message_final_dense(message)
        return tensor.update({'edge': {'message': message}})

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Aggregates messages.

        This method may be overridden by subclass.

        Arguments:
            tensor:
                A `GraphTensor` instance containing a message.
        """
        previous = tensor.node['feature']
        aggregate = tensor.aggregate('message', mode='mean')
        aggregate = keras.ops.concatenate([aggregate, previous], axis=-1)
        return tensor.update(
            {
                'node': {
                    'aggregate': aggregate, 
                },
                'edge': {
                    'message': None,
                }
            }
        )

    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Updates nodes. 

        This method may be overridden by subclass.

        Arguments:
            tensor:
                A `GraphTensor` instance containing aggregated messages 
                (updated node features).
        """
        aggregate = tensor.node['aggregate']
        node_feature = self._update_intermediate_dense(aggregate)
        node_feature = self._update_intermediate_activation(node_feature)
        node_feature = self._update_final_dense(node_feature)
        return tensor.update(
            {
                'node': {
                    'feature': node_feature,
                    'aggregate': None,
                },
            }
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self._activation),
            'normalize': self._normalize,
            'skip_connect': self._skip_connect,
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class GIConv(GraphConv):

    """Graph isomorphism network layer.

    >>> graph = molcraft.tensors.GraphTensor(
    ...     context={
    ...         'size': [2]
    ...     },
    ...     node={
    ...         'feature': [[1.], [2.]]
    ...     },
    ...     edge={
    ...         'source': [0, 1],
    ...         'target': [1, 0],
    ...     }
    ... )
    >>> conv = molcraft.layers.GIConv(units=4)
    >>> conv(graph)
        GraphTensor(
            context={
                'size': <tf.Tensor: shape=[1], dtype=int32>
            },
            node={
                'feature': <tf.Tensor: shape=[2, 4], dtype=float32>
            },
            edge={
                'source': <tf.Tensor: shape=[2], dtype=int32>,
                'target': <tf.Tensor: shape=[2], dtype=int32>
            }
        )
    """

    def __init__(
        self,
        units: int,
        activation: keras.layers.Activation | str | None = 'relu',
        use_bias: bool = True,
        normalize: bool = False,
        skip_connect: bool = True,
        update_edge_feature: bool = True,
        **kwargs,
    ):
        super().__init__(
            units=units, 
            activation=activation,
            use_bias=use_bias, 
            normalize=normalize, 
            skip_connect=skip_connect,
            **kwargs
        )
        self._update_edge_feature = update_edge_feature

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        node_feature_dim = spec.node['feature'].shape[-1]

        self.epsilon = self.add_weight(
            name='epsilon', 
            shape=(), 
            initializer='zeros',
            trainable=True,
        )

        if self.has_edge_feature:
            edge_feature_dim = spec.edge['feature'].shape[-1]

            if not self._update_edge_feature:
                if (edge_feature_dim != node_feature_dim):
                    warnings.warn(
                        'Found edge feature dim to be incompatible with node feature dim. '
                        'Automatically adding a edge feature projection layer to match '
                        'the dim of node features.',
                        stacklevel=2,
                    )
                    self._update_edge_feature = True 

            if self._update_edge_feature:
                self._edge_dense = self.get_dense(node_feature_dim)
        else:
            self._update_edge_feature = False

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        message = tensor.gather('feature', 'source')
        edge_feature = tensor.edge.get('feature')
        if self._update_edge_feature:
            edge_feature = self._edge_dense(edge_feature)
        if self.has_edge_feature:
            message += edge_feature
            message = keras.ops.relu(message)
        return tensor.update(
            {
                'edge': {
                    'message': message,
                    'feature': edge_feature
                }
            }
        )

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.aggregate('message', mode='mean')
        node_feature += (1 + self.epsilon) * tensor.node['feature']
        return tensor.update(
            {
                'node': {
                    'aggregate': node_feature,
                },
                'edge': {
                    'message': None,
                }
            }
        )
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'update_edge_feature': self._update_edge_feature
        })
        return config


@keras.saving.register_keras_serializable(package='molcraft')
class GAConv(GraphConv):

    """Graph attention network layer.

    >>> graph = molcraft.tensors.GraphTensor(
    ...     context={
    ...         'size': [2]
    ...     },
    ...     node={
    ...         'feature': [[1.], [2.]]
    ...     },
    ...     edge={
    ...         'source': [0, 1],
    ...         'target': [1, 0],
    ...     }
    ... )
    >>> conv = molcraft.layers.GAConv(units=4, heads=2)
    >>> conv(graph)
        GraphTensor(
            context={
                'size': <tf.Tensor: shape=[1], dtype=int32>
            },
            node={
                'feature': <tf.Tensor: shape=[2, 4], dtype=float32>
            },
            edge={
                'source': <tf.Tensor: shape=[2], dtype=int32>,
                'target': <tf.Tensor: shape=[2], dtype=int32>
            }
        )
    """

    def __init__(
        self,
        units: int,
        heads: int = 8,
        activation: keras.layers.Activation | str | None = "relu",
        use_bias: bool = True,
        normalize: bool = False,
        skip_connect: bool = True,
        update_edge_feature: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            units=units, 
            activation=activation,
            normalize=normalize, 
            use_bias=use_bias,
            skip_connect=skip_connect,
            **kwargs
        )
        self._heads = heads
        if self.units % self.heads != 0:
            raise ValueError(f"units need to be divisible by heads.")
        self._head_units = self.units // self.heads 
        self._update_edge_feature = update_edge_feature

    @property 
    def heads(self):
        return self._heads 
    
    @property 
    def head_units(self):
        return self._head_units 

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        self._update_edge_feature = self.has_edge_feature and self._update_edge_feature
        if self._update_edge_feature:
            self._edge_dense = self.get_einsum_dense(
                'ijh,jkh->ikh', (self.head_units, self.heads)
            )
        self._node_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._feature_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._attention_dense = self.get_einsum_dense(
            'ijh,jkh->ikh', (1, self.heads)
        )

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        attention_feature = keras.ops.concatenate(
            [
                tensor.gather('feature', 'source'),
                tensor.gather('feature', 'target')
            ], 
            axis=-1
        )
        if self.has_edge_feature:
            attention_feature = keras.ops.concatenate(
                [
                    attention_feature, 
                    tensor.edge['feature']
                ], 
                axis=-1
            )

        attention_feature = self._feature_dense(attention_feature)

        edge_feature = tensor.edge.get('feature')

        if self._update_edge_feature:
            edge_feature = self._edge_dense(attention_feature)
            edge_feature = keras.ops.reshape(edge_feature, (-1, self.units))
        
        attention_feature = keras.ops.leaky_relu(attention_feature)
        attention_score = self._attention_dense(attention_feature)
        attention_score = ops.edge_softmax(
            score=attention_score, edge_target=tensor.edge['target']
        )
        node_feature = self._node_dense(tensor.node['feature'])
        message = ops.gather(node_feature, tensor.edge['source'])
        message = ops.edge_weight(message, attention_score)
        return tensor.update(
            {
                'edge': {
                    'message': message,
                    'feature': edge_feature,
                }
            }
        )
    
    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.aggregate('message', mode='sum')
        node_feature = keras.ops.reshape(node_feature, (-1, self.units))
        return tensor.update(
            {
                'node': {
                    'aggregate': node_feature
                },
                'edge': {
                    'message': None,
                }
            }
        )
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "heads": self._heads,
            'update_edge_feature': self._update_edge_feature,
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class MPConv(GraphConv):

    """Message passing neural network layer.

    Also supports 3D molecular graphs.

    >>> graph = molcraft.tensors.GraphTensor(
    ...     context={
    ...         'size': [2]
    ...     },
    ...     node={
    ...         'feature': [[1.], [2.]]
    ...     },
    ...     edge={
    ...         'source': [0, 1],
    ...         'target': [1, 0],
    ...     }
    ... )
    >>> conv = molcraft.layers.MPConv(units=4)
    >>> conv(graph)
        GraphTensor(
            context={
                'size': <tf.Tensor: shape=[1], dtype=int32>
            },
            node={
                'feature': <tf.Tensor: shape=[2, 4], dtype=float32>
            },
            edge={
                'source': <tf.Tensor: shape=[2], dtype=int32>,
                'target': <tf.Tensor: shape=[2], dtype=int32>
            }
        )
    """

    def __init__(
        self, 
        units: int = 128, 
        activation: keras.layers.Activation | str | None = 'relu', 
        use_bias: bool = True,
        normalize: bool = False,
        skip_connect: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            units=units, 
            activation=activation,
            use_bias=use_bias,
            normalize=normalize, 
            skip_connect=skip_connect,
            **kwargs
        )

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        node_feature_dim = spec.node['feature'].shape[-1]
        self.update_fn = keras.layers.GRUCell(self.units)
        self._project_previous_node_feature = node_feature_dim != self.units
        if self._project_previous_node_feature:
            warnings.warn(
                'Input node feature dim does not match updated node feature dim. '
                'To make sure input node feature can be passed as `states` to the '
                'GRU cell, it will automatically be projected prior to it.',
                stacklevel=2
            )
            self._previous_node_dense = self.get_dense(self.units)

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Aggregates messages.

        This method may be overridden by subclass.

        Arguments:
            tensor:
                A `GraphTensor` instance containing a message.
        """
        aggregate = tensor.aggregate('message', mode='mean')
        return tensor.update(
            {
                'node': {
                    'aggregate': aggregate, 
                },
                'edge': {
                    'message': None,
                }
            }
        )
    
    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        previous = tensor.node['feature']
        aggregate = tensor.node['aggregate']
        if self._project_previous_node_feature:
            previous = self._previous_node_dense(previous)
        updated_node_feature, _ = self.update_fn(
            inputs=aggregate, states=previous
        )
        return tensor.update(
            {
                'node': {
                    'feature': updated_node_feature,
                    'aggregate': None,
                }
            }
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({})
        return config 
    

@keras.saving.register_keras_serializable(package='molcraft')
class GTConv(GraphConv):

    """Graph transformer layer.

    Also supports 3D molecular graphs.

    >>> graph = molcraft.tensors.GraphTensor(
    ...     context={
    ...         'size': [2]
    ...     },
    ...     node={
    ...         'feature': [[1.], [2.]]
    ...     },
    ...     edge={
    ...         'source': [0, 1],
    ...         'target': [1, 0],
    ...     }
    ... )
    >>> conv = molcraft.layers.GTConv(units=4, heads=2)
    >>> conv(graph)
        GraphTensor(
            context={
                'size': <tf.Tensor: shape=[1], dtype=int32>
            },
            node={
                'feature': <tf.Tensor: shape=[2, 4], dtype=float32>
            },
            edge={
                'source': <tf.Tensor: shape=[2], dtype=int32>,
                'target': <tf.Tensor: shape=[2], dtype=int32>,
            }
        )
    """

    def __init__(
        self,
        units: int,
        heads: int = 8,
        activation: keras.layers.Activation | str | None = "relu",
        use_bias: bool = True,
        normalize: bool = False,
        skip_connect: bool = True,
        attention_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            units=units, 
            activation=activation,
            normalize=normalize, 
            use_bias=use_bias,
            skip_connect=skip_connect,
            **kwargs
        )
        self._heads = heads
        if self.units % self.heads != 0:
            raise ValueError(f"units need to be divisible by heads.")
        self._head_units = self.units // self.heads 
        self._attention_dropout = attention_dropout

    @property 
    def heads(self):
        return self._heads 
    
    @property 
    def head_units(self):
        return self._head_units 
    
    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        self._query_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._key_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._value_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._output_dense = self.get_dense(self.units)
        self._softmax_dropout = keras.layers.Dropout(self._attention_dropout) 

        if self.has_edge_feature:
            self._attention_bias_dense_1 = self.get_einsum_dense('ij,jkh->ikh', (1, self.heads))

        if self.has_node_coordinate:
            node_feature_dim = spec.node['feature'].shape[-1]
            num_kernels = self.units
            self._gaussian_loc = self.add_weight(
                shape=[num_kernels], initializer='zeros', dtype='float32', trainable=True
            ) 
            self._gaussian_scale = self.add_weight(
                shape=[num_kernels], initializer='ones', dtype='float32', trainable=True
            )
            self._centrality_dense = self.get_dense(units=node_feature_dim)
            self._attention_bias_dense_2 = self.get_einsum_dense('ij,jkh->ikh', (1, self.heads))

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.node['feature']

        if self.has_node_coordinate:
            euclidean_distance = ops.euclidean_distance(
                tensor.gather('coordinate', 'target'),
                tensor.gather('coordinate', 'source'),
                axis=-1
            )
            gaussian = ops.gaussian(
                euclidean_distance, self._gaussian_loc, self._gaussian_scale
            )
            centrality = keras.ops.segment_sum(gaussian, tensor.edge['target'], tensor.num_nodes)
            node_feature += self._centrality_dense(centrality)

        query = self._query_dense(node_feature)
        key = self._key_dense(node_feature)
        value = self._value_dense(node_feature)

        query = ops.gather(query, tensor.edge['source'])
        key = ops.gather(key, tensor.edge['target'])
        value = ops.gather(value, tensor.edge['source'])

        attention_score = keras.ops.sum(query * key, axis=1, keepdims=True)
        attention_score /= keras.ops.sqrt(float(self.head_units))
            
        if self.has_edge_feature:
            attention_score += self._attention_bias_dense_1(tensor.edge['feature'])

        if self.has_node_coordinate:
            attention_score += self._attention_bias_dense_2(gaussian)

        attention = ops.edge_softmax(attention_score, tensor.edge['target'])
        attention = self._softmax_dropout(attention)

        if self.has_node_coordinate:
            displacement = ops.displacement(
                tensor.gather('coordinate', 'target'),
                tensor.gather('coordinate', 'source'),
                normalize=True
            )
            attention *= keras.ops.expand_dims(displacement, axis=-1)
            attention = keras.ops.expand_dims(attention, axis=2)
            value = keras.ops.expand_dims(value, axis=1)

        message = ops.edge_weight(value, attention)
        
        return tensor.update(
            {
                'edge': {
                    'message': message,
                },
            }
        )

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.aggregate('message', mode='sum')
        if self.has_node_coordinate:
            shape = (tensor.num_nodes, -1, self.units)
        else:
            shape = (tensor.num_nodes, self.units)
        node_feature = keras.ops.reshape(node_feature, shape)
        node_feature = self._output_dense(node_feature)
        if self.has_node_coordinate:
            node_feature = keras.ops.sum(node_feature, axis=1)
        return tensor.update(
            {
                'node': {
                    'aggregate': node_feature,
                },
                'edge': {
                    'message': None,
                }
            }
        )
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "heads": self._heads,
            'attention_dropout': self._attention_dropout,
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class EGConv(GraphConv):

    """Equivariant graph neural network layer 3D.

    Only supports 3D molecular graphs.

    >>> graph = molcraft.tensors.GraphTensor(
    ...     context={
    ...         'size': [2]
    ...     },
    ...     node={
    ...         'feature': [[1.], [2.]],
    ...         'coordinate': [[0.1, -0.1, 0.5], [1.2, -0.5, 2.1]],
    ...     },
    ...     edge={
    ...         'source': [0, 1],
    ...         'target': [1, 0],
    ...     }
    ... )
    >>> conv = molcraft.layers.EGConv(units=4)
    >>> conv(graph)
        GraphTensor(
            context={
                'size': <tf.Tensor: shape=[1], dtype=int32>
            },
            node={
                'feature': <tf.Tensor: shape=[2, 4], dtype=float32>,
                'coordinate': <tf.Tensor: shape=[2, 3], dtype=float32>
            },
            edge={
                'source': <tf.Tensor: shape=[2], dtype=int32>,
                'target': <tf.Tensor: shape=[2], dtype=int32>
            }
        )
    """

    def __init__(
        self, 
        units: int = 128, 
        activation: keras.layers.Activation | str | None = 'silu', 
        use_bias: bool = True,
        normalize: bool = False,
        skip_connect: bool = True,
        **kwargs
    ) -> None:
        super().__init__(
            units=units, 
            activation=activation,
            use_bias=use_bias,
            normalize=normalize, 
            skip_connect=skip_connect,
            **kwargs
        )

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        if not self.has_node_coordinate:
            raise ValueError(
                'Could not find `coordinate`s in node, '
                'which is required for Conv3D layers.'
            )
        self._message_feedforward_intermediate = self.get_dense(
            self.units, activation=self.activation
        )
        self._message_feedforward_final = self.get_dense(
            self.units, activation=self.activation
        )

        self._coord_feedforward_intermediate = self.get_dense(
            self.units, activation=self.activation
        ) 
        self._coord_feedforward_final = self.get_dense(
            1, use_bias=False, activation='tanh'
        )

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        relative_node_coordinate = keras.ops.subtract(
            tensor.gather('coordinate', 'target'), 
            tensor.gather('coordinate', 'source')
        ) 
        squared_distance = keras.ops.sum(
            keras.ops.square(relative_node_coordinate), 
            axis=-1, 
            keepdims=True
        )
    
        # For numerical stability (i.e., to prevent NaN losses), this implementation of `EGConv3D` 
        # either needs to apply a `tanh` activation to the output of `self._coord_feedforward_final`, 
        # or normalize `relative_node_cordinate` as follows:
        #
        # norm = keras.ops.sqrt(squared_distance) + keras.backend.epsilon()
        # relative_node_coordinate /= norm
        #
        # For now, this implementation does the former.

        feature = keras.ops.concatenate(
            [
                tensor.gather('feature', 'target'), 
                tensor.gather('feature', 'source'), 
                squared_distance, 
            ], 
            axis=-1
        )
        if self.has_edge_feature:
            feature = keras.ops.concatenate(
                [
                    feature,
                    tensor.edge['feature']
                ], 
                axis=-1
            )
        message = self._message_feedforward_final(
            self._message_feedforward_intermediate(feature)
        )

        relative_node_coordinate = keras.ops.multiply(
            relative_node_coordinate, 
            self._coord_feedforward_final(
                self._coord_feedforward_intermediate(message)
            )
        )
        return tensor.update(
            {
                'edge': {
                    'message': message,
                    'relative_node_coordinate': relative_node_coordinate
                }
            }
        )

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        coordinate = tensor.node['coordinate']
        coordinate += tensor.aggregate('relative_node_coordinate', mode='mean')

        # Original implementation seems to apply sum aggregation, which does not
        # seem work well for this implementation of `EGConv3D`, as it causes 
        # large output values and large initial losses. The magnitude of the 
        # aggregated values of a sum aggregation depends on the number of 
        # neighbors, which may be many and may differ from node to node (or 
        # graph to graph). Therefore, a mean mean aggregation is performed 
        # instead:
        aggregate = tensor.aggregate('message', mode='mean')
        aggregate = keras.ops.concatenate([aggregate, tensor.node['feature']], axis=-1)
        # Simply added to silence warning ('no gradients for variables ...')
        aggregate += (0.0 * keras.ops.sum(coordinate))

        return tensor.update(
            {
                'node': {
                    'aggregate': aggregate, 
                    'coordinate': coordinate,
                },
                'edge': {
                    'message': None,
                    'relative_node_coordinate': None
                }
            }
        ) 
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({})
        return config 
    

@keras.saving.register_keras_serializable(package='molcraft')
class Readout(GraphLayer):

    """Readout layer.
    """

    def __init__(self, mode: str | None = None, **kwargs):
        kwargs['kernel_initializer'] = None 
        kwargs['bias_initializer'] = None
        super().__init__(**kwargs)
        self.mode = mode
        if str(self.mode).lower().startswith('sum'):
            self._reduce_fn = keras.ops.segment_sum
        elif str(self.mode).lower().startswith('max'):
            self._reduce_fn = keras.ops.segment_max 
        elif str(self.mode).lower().startswith('super'):
            self._reduce_fn = keras.ops.segment_sum
        else:
            self._reduce_fn = ops.segment_mean

    def propagate(self, tensor: tensors.GraphTensor) -> tf.Tensor:
        node_feature = tensor.node['feature']
        if str(self.mode).lower().startswith('super'):
            node_feature = keras.ops.where(
                tensor.node['super'][:, None], node_feature, 0.0
            )
        return self._reduce_fn(
            node_feature, tensor.graph_indicator, tensor.num_subgraphs
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config['mode'] = self.mode 
        return config 
    

@keras.saving.register_keras_serializable(package='molcraft')
class NodeEmbedding(GraphLayer):

    """Node embedding layer.

    Embeds nodes based on its initial features.
    """

    def __init__(
        self, 
        dim: int = None, 
        normalize: bool = False,
        embed_context: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self._normalize = normalize
        self._embed_context = embed_context

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        feature_dim = spec.node['feature'].shape[-1]
        if not self.dim:
            self.dim = feature_dim
        self._node_dense = self.get_dense(self.dim)

        self._has_super = 'super' in spec.node
        has_context_feature = 'feature' in spec.context
        if not has_context_feature:
            self._embed_context = False 
        if self._has_super and not self._embed_context:
            self._super_feature = self.get_weight(shape=[self.dim], name='super_node_feature')
        if self._embed_context:
            self._context_dense = self.get_dense(self.dim)
        
        if not self._normalize:
            self._norm = keras.layers.Identity()
        elif str(self._normalize).lower().startswith('layer'):
            self._norm = keras.layers.LayerNormalization()
        else:
            self._norm = keras.layers.BatchNormalization()

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        feature = self._node_dense(tensor.node['feature'])

        if self._has_super and not self._embed_context:
            super_mask = keras.ops.expand_dims(tensor.node['super'], 1)
            feature = keras.ops.where(super_mask, self._super_feature, feature)

        if self._embed_context:
            context_feature = self._context_dense(tensor.context['feature'])
            feature = ops.scatter_update(feature, tensor.node['super'], context_feature)
            tensor = tensor.update({'context': {'feature': None}})

        feature = self._norm(feature)

        return tensor.update({'node': {'feature': feature}})

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'normalize': self._normalize,
            'embed_context': self._embed_context,
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class EdgeEmbedding(GraphLayer):

    """Edge embedding layer.

    Embeds edges based on its initial features.
    """

    def __init__(
        self, 
        dim: int = None, 
        normalize: bool = False,
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self._normalize = normalize

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        feature_dim = spec.edge['feature'].shape[-1]
        if not self.dim:
            self.dim = feature_dim
        self._edge_dense = self.get_dense(self.dim) 

        self._self_loop_feature = self.get_weight(shape=[self.dim], name='self_loop_edge_feature')

        self._has_super = 'super' in spec.edge
        if self._has_super:
            self._super_feature = self.get_weight(shape=[self.dim], name='super_edge_feature')
        
        if not self._normalize:
            self._norm = keras.layers.Identity()
        elif str(self._normalize).lower().startswith('layer'):
            self._norm = keras.layers.LayerNormalization()
        else:
            self._norm = keras.layers.BatchNormalization()

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        feature = self._edge_dense(tensor.edge['feature'])

        if self._has_super:
            super_mask = keras.ops.expand_dims(tensor.edge['super'], 1)
            feature = keras.ops.where(super_mask, self._super_feature, feature)

        self_loop_mask = keras.ops.expand_dims(tensor.edge['source'] == tensor.edge['target'], 1)
        feature = keras.ops.where(self_loop_mask, self._self_loop_feature, feature)

        feature = self._norm(feature)

        return tensor.update({'edge': {'feature': feature}})

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'normalize': self._normalize,
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class GraphNetwork(GraphLayer):

    """Graph neural network.

    Sequentially calls graph layers (`GraphLayer`) and concatenates its output. 

    Arguments:
        layers (list):
            A list of graph layers.
    """

    def __init__(self, layers: list[GraphLayer], **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = layers
        self._update_edge_feature = False

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        units = self.layers[0].units 
        node_feature_dim = spec.node['feature'].shape[-1]
        self._update_node_feature = node_feature_dim != units 
        if self._update_node_feature:
            warnings.warn(
                'Node feature dim does not match `units` of the first layer. '
                'Automatically adding a node projection layer to match `units`.',
                stacklevel=2
            )
            self._node_dense = self.get_dense(units)
        self._has_edge_feature = 'feature' in spec.edge 
        if self._has_edge_feature:
            edge_feature_dim = spec.edge['feature'].shape[-1]
            self._update_edge_feature = edge_feature_dim != units
            if self._update_edge_feature:
                warnings.warn(
                    'Edge feature dim does not match `units` of the first layer. '
                    'Automatically adding a edge projection layer to match `units`.',
                    stacklevel=2
                )
                self._edge_dense = self.get_dense(units)

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        x = tensors.to_dict(tensor)
        if self._update_node_feature:
            x['node']['feature'] = self._node_dense(tensor.node['feature'])
        if self._has_edge_feature and self._update_edge_feature:
            x['edge']['feature'] = self._edge_dense(tensor.edge['feature'])
        outputs = [x['node']['feature']]
        for layer in self.layers:
            x = layer(x)
            outputs.append(x['node']['feature'])
        return tensor.update(
            {
                'node': {
                    'feature': keras.ops.concatenate(outputs, axis=-1)
                }    
            }
        )
    
    def tape_propagate(
        self,
        tensor: tensors.GraphTensor,
        tape: tf.GradientTape,
        training: bool | None = None,
    ) -> tuple[tensors.GraphTensor, list[tf.Tensor]]:
        """Performs the propagation with a `GradientTape`.

        Performs the same forward pass as `propagate` but with a `GradientTape`
        watching intermediate node features.

        Arguments:
            tensor (tensors.GraphTensor):
                The graph input.
        """
        if isinstance(tensor, tensors.GraphTensor):
            x = tensors.to_dict(tensor)
        else:
            x = tensor
        if self._update_node_feature:
            x['node']['feature'] = self._node_dense(tensor.node['feature'])
        if self._update_edge_feature:
            x['edge']['feature'] = self._edge_dense(tensor.edge['feature'])
        tape.watch(x['node']['feature'])
        outputs = [x['node']['feature']]
        for layer in self.layers:
            x = layer(x, training=training)
            tape.watch(x['node']['feature'])
            outputs.append(x['node']['feature'])

        tensor = tensor.update(
            {
                'node': {
                    'feature': keras.ops.concatenate(outputs, axis=-1)
                }
            }
        )
        return tensor, outputs
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update(
            {
                'layers': [
                    keras.layers.serialize(layer) for layer in self.layers
                ]
            }
        )
        return config

    @classmethod
    def from_config(cls, config: dict) -> 'GraphNetwork':
        config['layers'] = [
            keras.layers.deserialize(layer) for layer in config['layers']
        ]
        return super().from_config(config)
    

@keras.saving.register_keras_serializable(package='molcraft')
class GaussianParams(keras.layers.Dense):
    '''Gaussian parameters.

    Computes loc and scale via a dense layer. Should be implemented
    as the last layer in a model and paired with `losses.GaussianNLL`.

    The loc and scale parameters (resulting from this layer) are concatenated
    together along the last axis, resulting in a single output tensor. 

    Args:
        events (int):
            The number of events. If the model makes a single prediction per example,
            then the number of events should be 1. If the model makes multiple predictions 
            per example, then the number of events should be greater than 1. 
            Default to 1.
        kwargs:
            See `keras.layers.Dense` documentation. `activation` will be applied
            to `loc` only. `scale` is automatically softplus activated.
    '''
    def __init__(self, events: int = 1, **kwargs):
        units = kwargs.pop('units', None)
        activation = kwargs.pop('activation', None)
        if units:
            if units % 2 != 0:
                raise ValueError(
                    '`units` needs to be divisble by 2 as `units` = 2 x `events`.'
                )
        else:
            units = int(events * 2)
        super().__init__(units=units, **kwargs)
        self.events = events
        self.loc_activation = keras.activations.get(activation)

    def call(self, inputs, **kwargs):
        loc_and_scale = super().call(inputs, **kwargs)
        loc = loc_and_scale[..., :self.events]
        scale = loc_and_scale[..., self.events:]
        scale = keras.ops.softplus(scale) + keras.backend.epsilon()
        loc = self.loc_activation(loc)
        return keras.ops.concatenate([loc, scale], axis=-1)

    def get_config(self):
        config = super().get_config()
        config['events'] = self.events
        config['units'] = None
        config['activation'] = keras.activations.serialize(self.loc_activation)
        return config
    

def Input(spec: tensors.GraphTensor.Spec) -> dict:
    """Used to specify inputs to model.

    Example:

    >>> import molcraft 
    >>> import keras
    >>> 
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer()
    >>> graph = featurizer([('N[C@@H](C)C(=O)O', 1.0), ('N[C@@H](CS)C(=O)O', 2.0)])
    >>> 
    >>> model = molcraft.models.GraphModel.from_layers(
    ...     molcraft.layers.Input(graph.spec),
    ...     molcraft.layers.NodeEmbedding(128),
    ...     molcraft.layers.EdgeEmbedding(128),
    ...     molcraft.layers.GraphTransformer(128),
    ...     molcraft.layers.GraphTransformer(128),
    ...     molcraft.layers.Readout('mean'),
    ...     molcraft.layers.Dense(1)
    ... ])
    """
    
    # Currently, Keras (3.8.0) does not support extension types.
    # So for now, this function will unpack the `GraphTensor.Spec` and 
    # return a dictionary of nested tensor specs. However, the corresponding 
    # nest of tensors will temporarily be converted to a `GraphTensor` by the 
    # `GraphLayer`, to levarage the utility of a `GraphTensor` object. 
    inputs = {}
    for outer_field, data in spec.__dict__.items():
        inputs[outer_field] = {}
        for inner_field, nested_spec in data.items():
            if outer_field == 'context' and inner_field in ['label', 'weight']:
                # Remove context label and weight from the symbolic input 
                # as a functional model is strict for what input can be passed.
                # (We want to train and predict with the model.)
                continue
            kwargs = {
                'shape': nested_spec.shape[1:],
                'dtype': nested_spec.dtype,
                'name': f'{outer_field}_{inner_field}'
            }
            if isinstance(nested_spec, tf.RaggedTensorSpec):
                kwargs['ragged'] = True
            try:
                inputs[outer_field][inner_field] = keras.Input(**kwargs)
            except TypeError:
                raise ValueError(
                    "`keras.Input` does not currently support ragged tensors. For now, "
                    "pass the `Spec` of a 'flat' `GraphTensor` to `GNNInput`." 
                )
    return inputs


def _serialize_spec(spec: tensors.GraphTensor.Spec) -> dict:
    serialized_spec = {}
    for outer_field, data in spec.__dict__.items():
        serialized_spec[outer_field] = {}
        for inner_field, inner_spec in data.items():
            serialized_spec[outer_field][inner_field] = {
                'shape': inner_spec.shape.as_list(), 
                'dtype': inner_spec.dtype.name, 
                'name': inner_spec.name,
            }
    return serialized_spec

def _deserialize_spec(serialized_spec: dict) -> tensors.GraphTensor.Spec:
    deserialized_spec = {}
    for outer_field, data in serialized_spec.items():
        deserialized_spec[outer_field] = {}
        for inner_field, inner_spec in data.items():
            deserialized_spec[outer_field][inner_field] = tf.TensorSpec(
                inner_spec['shape'], inner_spec['dtype'], inner_spec['name']
            )
    return tensors.GraphTensor.Spec(**deserialized_spec)

def _spec_from_inputs(inputs):
    symbolic_inputs = keras.backend.is_keras_tensor(
        tf.nest.flatten(inputs)[0]
    )
    if not symbolic_inputs:
        nested_specs = tf.nest.map_structure(
            tf.type_spec_from_value, inputs
        )
    else:
        nested_specs = tf.nest.map_structure(
            lambda t: tf.TensorSpec(t.shape, t.dtype), inputs
        )
    if isinstance(nested_specs, tensors.GraphTensor.Spec):
        spec = nested_specs
        return spec
    return tensors.GraphTensor.Spec(**nested_specs)


GraphTransformer = GTConv
