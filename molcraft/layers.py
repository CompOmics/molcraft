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

    def __call__(self, inputs, **kwargs):
        if not self.built:
            spec = _spec_from_inputs(inputs)
            self.build(spec)
        convert = isinstance(inputs, tensors.GraphTensor)
        if convert:
            inputs = tensors.to_dict(inputs)
        if isinstance(self, functional.Functional):
            inputs, left_out_inputs = _match_functional_input(self.input, inputs)
        outputs = super().__call__(inputs, **kwargs)
        if not tensors.is_graph(outputs):
            return outputs
        if isinstance(self, functional.Functional):
            outputs = _add_left_out_inputs(outputs, left_out_inputs)
        if convert:
            outputs = tensors.from_dict(outputs)
        return outputs

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
            Activation function to use. If not specified, a linear activation (a(x) = x) is used.
            Default to `None`.
        use_bias (bool):
            Whether bias should be used in dense layers. Default to `True`.
        normalization (bool, str):
            Whether `LayerNormalization` should be applied to the final node feature output.
            To use `BatchNormalization`, specify `batch_norm`. Default to `False`.
        skip_connection (bool, str):
            Whether node feature input should be added to the node feature output. 
            If node feature input dim is not equal to `units` (node feature output dim), 
            a projection layer will automatically project the residual before adding it 
            to the output. To use weighted skip connection, 
            specify `weighted`. The weight multiplied with the skip connection is a 
            learnable scalar. Default to `True`.
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
        activation: str | keras.layers.Activation | None = None,
        use_bias: bool = True,
        normalization: bool | str = False,
        skip_connection: bool | str = True, 
        **kwargs
    ) -> None:
        super().__init__(use_bias=use_bias, **kwargs)
        self._units = units
        self._normalization = normalization
        self._skip_connection = skip_connection
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
    
    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Forward pass.

        Invokes `message(graph)`, `aggregate(graph)` and `update(graph)` in sequence.

        Arguments:
            tensor:
                A `GraphTensor` instance.
        """
        if self._skip_connection:
            input_node_feature = tensor.node['feature']
            if self._project_input_node_feature:
                input_node_feature = self._residual_projection(input_node_feature)

        tensor = self.message(tensor)
        tensor = self.aggregate(tensor)
        tensor = self.update(tensor)

        updated_node_feature = tensor.node['feature']

        if self._skip_connection:
            if self._use_weighted_skip_connection:
                input_node_feature *= self._skip_connection_weight
            updated_node_feature += input_node_feature
        
        if self._normalization:
            updated_node_feature = self._output_norm(updated_node_feature)

        return tensor.update({'node': {'feature': updated_node_feature}})
    
    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        if not self.units:
            raise ValueError(
                f'`self.units` needs to be a positive integer. Found: {self.units}.'
            )
        node_feature_dim = spec.node['feature'].shape[-1]
        self._project_input_node_feature = (
            self._skip_connection and (node_feature_dim != self.units)
        )
        if self._project_input_node_feature:
            warn(
                '`skip_connection` is set to `True`, but found incompatible dim ' 
                'between input (node feature dim) and output (`self.units`). '
                'Automatically applying a projection layer to residual to '
                'match input and output. '
            )
            self._residual_projection = self.get_dense(
                self.units, name='residual_projection'
            )

        skip_connection = str(self._skip_connection).lower()
        self._use_weighted_skip_connection = skip_connection.startswith('weight')
        if self._use_weighted_skip_connection:
            self._skip_connection_weight = self.add_weight(
                name='skip_connection_weight', 
                shape=(), 
                initializer='ones',
                trainable=True,
            )

        if self._normalization:
            if str(self._normalization).lower().startswith('batch'):
                self._output_norm = keras.layers.BatchNormalization(
                    name='output_batch_norm'
                )
            else:
                self._output_norm = keras.layers.LayerNormalization(
                    name='output_layer_norm'
                )

        self._has_edge_feature = 'edge' in spec.edge 

        has_overridden_message = self.__class__.message != GraphConv.message 
        if not has_overridden_message:
            self._message_dense = self.get_dense(self.units)

        has_overridden_update = self.__class__.update != GraphConv.update 
        if not has_overridden_update:
            self._output_dense = self.get_dense(self.units)
            self._output_activation = self._activation

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Compute messages.

        This method may be overridden by subclass.

        Arguments:
            tensor:
                The inputted `GraphTensor` instance.
        """
        if not self._has_edge_feature:
            message_feature = tensor.gather('feature', 'source')
        else:
            message_feature = keras.ops.concatenate(
                [
                    tensor.gather('feature', 'source'),
                    tensor.edge['feature']
                ], 
                axis=-1
            )
        message = self._message_dense(message_feature)
        return tensor.update(
            {
                'edge': {
                    'message': message
                }
            }
        )

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
                    'feature': aggregate,
                    'previous_feature': tensor.node['feature']
                },
                'edge': {
                    'message': None
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
        if not 'previous_feature' in tensor.node:
            feature = tensor.node['feature']
        else:
            feature = keras.ops.concatenate(
                [
                    tensor.node['feature'],
                    tensor.node['previous_feature']
                ],
                axis=-1
            )
        update = self._output_dense(feature)
        update = self._output_activation(update)
        return tensor.update(
            {
                'node': {
                    'feature': update,
                    'previous_feature': None,
                }
            }
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self._activation),
            'normalization': self._normalization,
            'skip_connection': self._skip_connection,
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
        normalization: bool = False,
        update_edge_feature: bool = True,
        **kwargs,
    ):
        super().__init__(
            units=units, 
            activation=activation,
            normalization=normalization, 
            use_bias=use_bias, 
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

        self._has_edge_feature = 'feature' in spec.edge
        if self._has_edge_feature:
            edge_feature_dim = spec.edge['feature'].shape[-1]

            if not self._update_edge_feature:
                if (edge_feature_dim != node_feature_dim):
                    warn(
                        'Found edge feature dim to be incompatible with node feature dim. '
                        'Automatically adding a edge feature projection layer to match '
                        'the dim of node features.'
                    )
                    self._update_edge_feature = True 

            if self._update_edge_feature:
                self._edge_dense = self.get_dense(node_feature_dim)
        else:
            self._update_edge_feature = False

        has_overridden_update = self.__class__.update != GIConv.update 
        if not has_overridden_update:
            self._feedforward_intermediate_dense = self.get_dense(self.units)
            self._feedforward_activation = self._activation
            self._feedforward_output_dense = self.get_dense(self.units)

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        message = tensor.gather('feature', 'source')
        edge_feature = tensor.edge.get('feature')
        if self._update_edge_feature:
            edge_feature = self._edge_dense(edge_feature)
        if self._has_edge_feature:
            message += edge_feature
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
                    'feature': node_feature,
                },
                'edge': {
                    'message': None,
                }
            }
        )
    
    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.node['feature']
        node_feature = self._feedforward_intermediate_dense(node_feature)
        node_feature = self._feedforward_activation(node_feature)
        node_feature = self._feedforward_output_dense(node_feature)
        return tensor.update(
            {
                'node': {
                    'feature': node_feature,
                }
            }
        )
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'update_edge_feature': self._update_edge_feature
        })
        return config


@keras.saving.register_keras_serializable(package='molgraphx')
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
        normalization: bool = False,
        update_edge_feature: bool = True,
        attention_activation: keras.layers.Activation | str | None = "leaky_relu",
        **kwargs,
    ) -> None:
        kwargs['skip_connection'] = False
        super().__init__(
            units=units, 
            activation=activation,
            use_bias=use_bias,
            normalization=normalization, 
            **kwargs
        )
        self._heads = heads
        if self.units % self.heads != 0:
            raise ValueError(f"units need to be divisible by heads.")
        self._head_units = self.units // self.heads 
        self._update_edge_feature = update_edge_feature
        self._attention_activation = keras.activations.get(attention_activation)

    @property 
    def heads(self):
        return self._heads 
    
    @property 
    def head_units(self):
        return self._head_units 

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        self._has_edge_feature = 'feature' in spec.edge 
        self._update_edge_feature = self._has_edge_feature and self._update_edge_feature
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
        self._node_self_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )

        has_overridden_update = self.__class__.update != GAConv.update 
        if not has_overridden_update:
            self._feedforward_intermediate_dense = self.get_dense(self.units)
            self._feedforward_activation = self._activation
            self._feedforward_output_dense = self.get_dense(self.units)

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        attention_feature = keras.ops.concatenate(
            [
                tensor.gather('feature', 'source'),
                tensor.gather('feature', 'target')
            ], 
            axis=-1
        )
        if self._has_edge_feature:
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
        
        attention_feature = self._attention_activation(attention_feature)
        attention_score = self._attention_dense(attention_feature)
        attention_score = ops.edge_softmax(
            score=attention_score, edge_target=tensor.edge['target']
        )
        node_feature = self._node_dense(tensor.node['feature'])
        message = ops.gather(node_feature, tensor.edge['source'])
        return tensor.update(
            {
                'edge': {
                    'message': message,
                    'weight': attention_score,
                    'feature': edge_feature,
                }
            }
        )
    
    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.aggregate('message', mode='sum')
        node_feature += self._node_self_dense(tensor.node['feature'])
        node_feature = keras.ops.reshape(node_feature, (-1, self.units))
        return tensor.update(
            {
                'node': {
                    'feature': node_feature
                },
                'edge': {
                    'message': None,
                    'weight': None,
                }
            }
        )
    
    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.node['feature']
        node_feature = self._feedforward_intermediate_dense(node_feature)
        node_feature = self._feedforward_activation(node_feature)
        node_feature = self._feedforward_output_dense(node_feature)
        return tensor.update(
            {
                'node': {
                    'feature': node_feature
                }
            }
        )
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            "heads": self._heads,
            'update_edge_feature': self._update_edge_feature,
            'attention_activation': keras.activations.serialize(self._attention_activation),
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class GTConv(GraphConv):

    """Graph transformer layer.

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
        normalization: bool = False,
        attention_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__(
            units=units, 
            activation=activation,
            use_bias=use_bias,
            normalization=normalization, 
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

        self._add_bias = not 'bias' in spec.edge

        if self._add_bias:
            self._edge_bias = EdgeBias(biases=self.heads)

        has_overridden_update = self.__class__.update != GTConv.update 
        if not has_overridden_update:
            self._feedforward_intermediate_dense = self.get_dense(self.units)
            self._feedforward_activation = self._activation
            self._feedforward_output_dense = self.get_dense(self.units)

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        if self._add_bias:
            edge_bias = self._edge_bias(tensor)
            tensor = tensor.update(
                {
                    'edge': {
                        'bias': edge_bias
                    }
                }
            )
            
        node_feature = tensor.node['feature']
        
        query = self._query_dense(node_feature)
        key = self._key_dense(node_feature)
        value = self._value_dense(node_feature)

        query = ops.gather(query, tensor.edge['source'])
        key = ops.gather(key, tensor.edge['target'])
        value = ops.gather(value, tensor.edge['source'])

        attention_score = keras.ops.sum(query * key, axis=1, keepdims=True)
        attention_score /= keras.ops.sqrt(float(self.head_units))
            
        attention_score += keras.ops.expand_dims(tensor.edge['bias'], axis=1)
        attention = ops.edge_softmax(attention_score, tensor.edge['target'])
        attention = self._softmax_dropout(attention)

        return tensor.update(
            {
                'edge': {
                    'message': value,
                    'weight': attention,
                },
            }
        )

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.aggregate('message', mode='sum')
        node_feature = keras.ops.reshape(node_feature, (-1, self.units))
        node_feature = self._output_dense(node_feature)
        return tensor.update(
            {
                'node': {
                    'feature': node_feature,
                    'residual': tensor.node['feature']
                },
                'edge': {
                    'message': None,
                    'weight': None,
                }
            }
        )

    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.node['feature']
        node_feature = self._feedforward_intermediate_dense(node_feature)
        node_feature = self._feedforward_activation(node_feature)
        node_feature = self._feedforward_output_dense(node_feature)
        return tensor.update(
            {
                'node': {
                    'feature': node_feature,
                },
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
class MPConv(GraphConv):

    """Message passing neural network layer.
    """

    def __init__(
        self, 
        units: int = 128, 
        activation: keras.layers.Activation | str | None = None, 
        use_bias: bool = True,
        normalization: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            units=units, 
            activation=activation,
            use_bias=use_bias,
            normalization=normalization, 
            **kwargs
        )

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        node_feature_dim = spec.node['feature'].shape[-1]
        self.message_fn = self.get_dense(self.units, activation=self._activation)
        self.update_fn = keras.layers.GRUCell(self.units)
        self._has_edge_feature = 'feature' in spec.edge
        self.project_input_node_feature = node_feature_dim != self.units
        if self.project_input_node_feature:
            warn(
                'Input node feature dim does not match updated node feature dim. '
                'To make sure input node feature can be passed as `states` to the '
                'GRU cell, it will automatically be projected prior to it.'
            )
            self._previous_node_dense = self.get_dense(
                self.units, activation=self._activation
            )

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        feature = keras.ops.concatenate(
            [
                tensor.gather('feature', 'source'),
                tensor.gather('feature', 'target'),
            ], 
            axis=-1
        )
        if self._has_edge_feature:
            feature = keras.ops.concatenate(
                [
                    feature,
                    tensor.edge['feature']
                ], 
                axis=-1
            )
        message = self.message_fn(feature)
        return tensor.update(
            {
                'edge': {
                    'message': message,
                }
            }
        )

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        aggregate = tensor.aggregate('message', mode='mean')
        previous = tensor.node['feature']
        if self.project_input_node_feature:
            previous = self._previous_node_dense(previous)
        return tensor.update(
            {
                'node': {
                    'feature': aggregate,
                    'previous_feature': previous,
                }
            }
        )

    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        updated_node_feature, _ = self.update_fn(
            inputs=tensor.node['feature'],
            states=tensor.node['previous_feature']
        )
        return tensor.update(
            {
                'node': {
                    'feature': updated_node_feature,
                    'previous_feature': None,
                }
            }
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({})
        return config 
    

@keras.saving.register_keras_serializable(package='molcraft')
class GTConv3D(GTConv):

    """Graph transformer layer 3D.
    """

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        super().build(spec)
        if self._add_bias:
            node_feature_dim = spec.node['feature'].shape[-1]
            kernels = self.units
            self._gaussian_basis = GaussianDistance(kernels)
            self._centrality_dense = self.get_dense(units=node_feature_dim)
            self._gaussian_edge_bias = self.get_dense(self.heads)

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.node['feature']

        if self._add_bias:
            gaussian = self._gaussian_basis(tensor)
            centrality = keras.ops.segment_sum(
                gaussian, tensor.edge['target'], tensor.num_nodes
            )
            node_feature += self._centrality_dense(centrality)

            edge_bias = self._edge_bias(tensor) + self._gaussian_edge_bias(gaussian)
            tensor = tensor.update({'edge': {'bias': edge_bias}})
            
        query = self._query_dense(node_feature)
        key = self._key_dense(node_feature)
        value = self._value_dense(node_feature)

        query = ops.gather(query, tensor.edge['source'])
        key = ops.gather(key, tensor.edge['target'])
        value = ops.gather(value, tensor.edge['source'])

        attention_score = keras.ops.sum(query * key, axis=1, keepdims=True)
        attention_score /= keras.ops.sqrt(float(self.head_units))
            
        attention_score += keras.ops.expand_dims(tensor.edge['bias'], axis=1)
        
        attention = ops.edge_softmax(attention_score, tensor.edge['target'])
        attention = self._softmax_dropout(attention)

        distance = keras.ops.subtract(
            tensor.gather('coordinate', 'source'),
            tensor.gather('coordinate', 'target')
        )
        euclidean_distance = ops.euclidean_distance(
            tensor.gather('coordinate', 'source'),
            tensor.gather('coordinate', 'target'),
            axis=-1
        )
        distance /= euclidean_distance

        attention *= keras.ops.expand_dims(distance, axis=-1)
        attention = keras.ops.expand_dims(attention, axis=2)
        value = keras.ops.expand_dims(value, axis=1)
        
        return tensor.update(
            {
                'edge': {
                    'message': value,
                    'weight': attention,
                },
            }
        )

    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        node_feature = tensor.aggregate('message', mode='sum')
        node_feature = keras.ops.reshape(
            node_feature, (tensor.num_nodes, -1, self.units)
        )
        node_feature = self._output_dense(node_feature)
        node_feature = keras.ops.sum(node_feature, axis=1)
        return tensor.update(
            {
                'node': {
                    'feature': node_feature,
                    'residual': tensor.node['feature']
                },
                'edge': {
                    'message': None,
                    'weight': None,
                }
            }
        )
    

@keras.saving.register_keras_serializable(package='molcraft')
class MPConv3D(MPConv):

    """Message passing neural network layer 3D.
    """
    
    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        euclidean_distance = ops.euclidean_distance(
            tensor.gather('coordinate', 'target'),
            tensor.gather('coordinate', 'source'),
            axis=-1
        )
        feature = keras.ops.concatenate(
            [
                tensor.gather('feature', 'source'),
                tensor.gather('feature', 'target'),
                euclidean_distance,
            ], 
            axis=-1
        )
        if self._has_edge_feature:
            feature = keras.ops.concatenate(
                [
                    feature,
                    tensor.edge['feature']
                ], 
                axis=-1
            )
        message = self.message_fn(feature)
        return tensor.update(
            {
                'edge': {
                    'message': message,
                }
            }
        )
    

@keras.saving.register_keras_serializable(package='molcraft')
class EGConv3D(GraphConv):

    """Equivariant graph neural network layer 3D.
    """

    def __init__(
        self, 
        units: int = 128, 
        activation: keras.layers.Activation | str | None = None, 
        use_bias: bool = True,
        normalization: bool = False,
        **kwargs
    ) -> None:
        super().__init__(
            units=units, 
            activation=activation,
            use_bias=use_bias,
            normalization=normalization, 
            **kwargs
        )

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        if 'coordinate' not in spec.node:
            raise ValueError(
                'Could not find `coordinate`s in node, '
                'which is required for Conv3D layers.'
            )
        self._has_edge_feature = 'feature' in spec.edge
        self.message_fn = self.get_dense(self.units, activation=self._activation)
        self.dense_position = self.get_dense(1)

        has_overridden_update = self.__class__.update != EGConv3D.update 
        if not has_overridden_update:
            self.update_fn = self.get_dense(self.units, activation=self._activation)
            self.output_dense = self.get_dense(self.units)

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        relative_node_coordinate = keras.ops.subtract(
            tensor.gather('coordinate', 'target'), 
            tensor.gather('coordinate', 'source')
        ) 
        euclidean_distance = keras.ops.sum(
            keras.ops.square(
                relative_node_coordinate
            ), 
            axis=-1, 
            keepdims=True
        )
        feature = keras.ops.concatenate(
            [
                tensor.gather('feature', 'target'), 
                tensor.gather('feature', 'source'), 
                euclidean_distance, 
            ], 
            axis=-1
        )
        if self._has_edge_feature:
            feature = keras.ops.concatenate(
                [
                    feature,
                    tensor.edge['feature']
                ], 
                axis=-1
            )
        message = self.message_fn(feature)
        relative_node_coordinate = keras.ops.multiply(
            relative_node_coordinate,
            self.dense_position(message)
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
        coefficient = keras.ops.bincount(
            tensor.edge['source'], 
            minlength=tensor.num_nodes
        )
        coefficient = keras.ops.cast(
            coefficient, tensor.node['coordinate'].dtype
        )
        coefficient = keras.ops.expand_dims(
            keras.ops.divide_no_nan(1, coefficient), axis=1
        )  

        updated_coordinate = tensor.aggregate('relative_node_coordinate') * coefficient
        updated_coordinate += tensor.node['coordinate']

        aggregate = tensor.aggregate('message', mode='mean')
        return tensor.update(
            {
                'node': {
                    'feature': aggregate, 
                    'coordinate': updated_coordinate,
                    'previous_feature': tensor.node['feature'],
                },
                'edge': {
                    'message': None,
                    'relative_node_coordinate': None
                }
            }
        ) 

    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        updated_node_feature = self.update_fn(
            keras.ops.concatenate(
                [
                    tensor.node['feature'], 
                    tensor.node['previous_feature']
                ], 
                axis=-1
            )
        )  
        updated_node_feature = self.output_dense(updated_node_feature)
        return tensor.update(
            {
                'node': {
                    'feature': updated_node_feature, 
                    'previous_feature': None,
                },
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
            warn(
                'Node feature dim does not match `units` of the first layer. '
                'Automatically adding a node projection layer to match `units`.'
            )
            self._node_dense = self.get_dense(units)
        self._has_edge_feature = 'feature' in spec.edge 
        if self._has_edge_feature:
            edge_feature_dim = spec.edge['feature'].shape[-1]
            self._update_edge_feature = edge_feature_dim != units
            if self._update_edge_feature:
                warn(
                    'Edge feature dim does not match `units` of the first layer. '
                    'Automatically adding a edge projection layer to match `units`.'
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
class NodeEmbedding(GraphLayer):

    """Node embedding layer.

    Embeds nodes based on its initial features.
    """

    def __init__(
        self, 
        dim: int = None, 
        normalization: bool = False,
        embed_context: bool = True,
        allow_reconstruction: bool = False,
        allow_masking: bool = True, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self._normalization = normalization
        self._embed_context = embed_context
        self._masking_rate = None
        self._allow_masking = allow_masking
        self._allow_reconstruction = allow_reconstruction

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
        if self._allow_masking:
            self._mask_feature = self.get_weight(shape=[self.dim], name='mask_node_feature')

        if self._embed_context:
            context_feature_dim = spec.context['feature'].shape[-1]
            self._context_dense = self.get_dense(self.dim)
        
        if self._normalization:
            if str(self._normalization).lower().startswith('batch'):
                self._norm = keras.layers.BatchNormalization(
                    name='output_batch_norm'
                )
            else:
                self._norm = keras.layers.LayerNormalization(
                    name='output_layer_norm'
                )

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        feature = self._node_dense(tensor.node['feature'])

        if self._has_super:
            super_feature = (0 if self._embed_context else self._super_feature)
            super_mask = keras.ops.expand_dims(tensor.node['super'], 1)
            feature = keras.ops.where(super_mask, super_feature, feature)

        if self._embed_context:
            context_feature = self._context_dense(tensor.context['feature'])
            feature = ops.scatter_update(feature, tensor.node['super'], context_feature)
            tensor = tensor.update({'context': {'feature': None}})

        if (
            self._allow_masking and 
            self._masking_rate is not None and 
            self._masking_rate > 0
        ):
            random = keras.random.uniform(shape=[tensor.num_nodes])
            mask = random <= self._masking_rate
            if self._has_super:
                mask = keras.ops.logical_and(
                    mask, keras.ops.logical_not(tensor.node['super'])
                )
            mask = keras.ops.expand_dims(mask, -1)
            feature = keras.ops.where(mask, self._mask_feature, feature)
        elif self._allow_masking:
            # Slience warning of 'no gradients for variables'
            feature = feature + (self._mask_feature * 0.0)

        if self._normalization:
            feature = self._norm(feature)

        if not self._allow_reconstruction:
            return tensor.update({'node': {'feature': feature}})
        return tensor.update({'node': {'feature': feature, 'target_feature': feature}})
    
    @property 
    def masking_rate(self):
        return self._masking_rate 
    
    @masking_rate.setter
    def masking_rate(self, rate: float):
        if not self._allow_masking and rate is not None:
            raise ValueError(
                f'Cannot set `masking_rate` for layer {self} '
                'as `allow_masking` was set to `False`.'
            )
        self._masking_rate = float(rate)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'normalization': self._normalization,
            'embed_context': self._embed_context,
            'allow_masking': self._allow_masking,
            'allow_reconstruction': self._allow_reconstruction,
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
        normalization: bool = False,
        allow_masking: bool = True, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self._normalization = normalization
        self._masking_rate = None
        self._allow_masking = allow_masking

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        feature_dim = spec.edge['feature'].shape[-1]
        if not self.dim:
            self.dim = feature_dim
        self._edge_dense = self.get_dense(self.dim)

        self._has_super = 'super' in spec.edge
        if self._has_super:
            self._super_feature = self.get_weight(shape=[self.dim], name='super_edge_feature')
        if self._allow_masking:
            self._mask_feature = self.get_weight(shape=[self.dim], name='mask_edge_feature')

        if self._normalization:
            if str(self._normalization).lower().startswith('batch'):
                self._norm = keras.layers.BatchNormalization(
                    name='output_batch_norm'
                )
            else:
                self._norm = keras.layers.LayerNormalization(
                    name='output_layer_norm'
                )

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        feature = self._edge_dense(tensor.edge['feature'])

        if self._has_super:
            super_feature = self._super_feature
            super_mask = keras.ops.expand_dims(tensor.edge['super'], 1)
            feature = keras.ops.where(super_mask, super_feature, feature)

        if (
            self._allow_masking and 
            self._masking_rate is not None and 
            self._masking_rate > 0
        ):
            random = keras.random.uniform(shape=[tensor.num_edges])
            mask = random <= self._masking_rate
            if self._has_super:
                mask = keras.ops.logical_and(
                    mask, keras.ops.logical_not(tensor.edge['super'])
                )
            mask = keras.ops.expand_dims(mask, -1)
            feature = keras.ops.where(mask, self._mask_feature, feature)
        elif self._allow_masking:
            # Slience warning of 'no gradients for variables'
            feature = feature + (self._mask_feature * 0.0)

        if self._normalization:
            feature = self._norm(feature)

        return tensor.update({'edge': {'feature': feature, 'embedding': feature}})

    @property 
    def masking_rate(self):
        return self._masking_rate 
    
    @masking_rate.setter
    def masking_rate(self, rate: float):
        if not self._allow_masking and rate is not None:
            raise ValueError(
                f'Cannot set `masking_rate` for layer {self} '
                'as `allow_masking` was set to `False`.'
            )
        self._masking_rate = float(rate)

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'dim': self.dim,
            'normalization': self._normalization,
            'allow_masking': self._allow_masking
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class Projection(GraphLayer):
    """Base graph projection layer.
    """
    def __init__(
        self, 
        units: int = None, 
        activation: str | keras.layers.Activation | None = None, 
        use_bias: bool = True,
        field: str = 'node',
        **kwargs
    ) -> None:
        super().__init__(use_bias=use_bias, **kwargs)
        self.units = units
        self._activation = keras.activations.get(activation)
        self.field = field 

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        data = getattr(spec, self.field, None)
        if data is None:
            raise ValueError('Could not access field {self.field!r}.')
        feature_dim = data['feature'].shape[-1]
        if not self.units:
            self.units = feature_dim
        self._dense = self.get_dense(self.units)

    def propagate(self, tensor: tensors.GraphTensor):
        feature = getattr(tensor, self.field)['feature']
        feature = self._dense(feature)
        feature = self._activation(feature)
        return tensor.update(
            {
                self.field: {
                    'feature': feature
                }
            }
        ) 

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'units': self.units,
            'activation': keras.activations.serialize(self._activation),
            'field': self.field,
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class ContextProjection(Projection):
    """Context projection layer.
    """
    def __init__(self, units: int = None, activation: str = None, **kwargs):
        kwargs['field'] = 'context'
        super().__init__(units=units, activation=activation, **kwargs)


@keras.saving.register_keras_serializable(package='molcraft')
class NodeProjection(Projection):
    """Node projection layer.
    """
    def __init__(self, units: int = None, activation: str = None, **kwargs):
        kwargs['field'] = 'node'
        super().__init__(units=units, activation=activation, **kwargs)


@keras.saving.register_keras_serializable(package='molcraft')
class EdgeProjection(Projection):
    """Edge projection layer.
    """
    def __init__(self, units: int = None, activation: str = None, **kwargs):
        kwargs['field'] = 'edge'
        super().__init__(units=units, activation=activation, **kwargs)


@keras.saving.register_keras_serializable(package='molcraft')
class Reconstruction(GraphLayer):

    def __init__(
        self, 
        loss: keras.losses.Loss | str = 'mse',
        loss_weight: float = 0.5, 
        **kwargs
    ):
        super().__init__(**kwargs)
        self._loss_fn = keras.losses.get(loss)
        self._loss_weight = loss_weight
    
    def build(self, spec: tensors.GraphTensor.Spec) -> None:
         has_target_node_feature = 'target_feature' in spec.node
         if not has_target_node_feature:
             raise ValueError(
                'Could not find `target_feature` in `spec.node`. '
                'Add a `target_feature` via `NodeEmbedding` by setting '
                '`allow_reconstruction` to `True`.'
            )
         output_dim = spec.node['target_feature'].shape[-1]
         self._dense = self.get_dense(output_dim)

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        target_node_feature = tensor.node['target_feature']
        transformed_node_feature = tensor.node['feature']

        reconstructed_node_feature = self._dense(
            transformed_node_feature
        )

        loss = self._loss_fn(
            target_node_feature, reconstructed_node_feature
        ) 
        self.add_loss(keras.ops.sum(loss) * self._loss_weight)
        return tensor.update({'node': {'feature': transformed_node_feature}})
    
    def get_config(self):
        config = super().get_config()
        config['loss'] = keras.losses.serialize(self._loss_fn)
        config['loss_weight'] = self._loss_weight 
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class EdgeBias(GraphLayer):

    def __init__(self, biases: int, **kwargs):
        super().__init__(**kwargs)
        self.biases = biases

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        self._has_edge_length = 'length' in spec.edge
        self._has_edge_feature = 'feature' in spec.edge
        if self._has_edge_feature:
            self._edge_feature_dense = self.get_dense(self.biases)
        if self._has_edge_length:
            self._edge_length_dense = self.get_dense(
                self.biases, kernel_initializer='zeros'
            )

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        bias = keras.ops.zeros(
            shape=(tensor.num_edges, self.biases), 
            dtype=tensor.node['feature'].dtype
        )
        if self._has_edge_feature:
            bias += self._edge_feature_dense(tensor.edge['feature'])
        if self._has_edge_length:
            bias += self._edge_length_dense(tensor.edge['length'])
        return bias

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({'biases': self.biases})
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class GaussianDistance(GraphLayer):

    def __init__(self, kernels: int, **kwargs):
        super().__init__(**kwargs)
        self.kernels = kernels 

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
        self._loc = self.add_weight(
            shape=[self.kernels],
            initializer='zeros',
            dtype='float32',
            trainable=True
        ) 
        self._scale = self.add_weight(
            shape=[self.kernels],
            initializer='ones',
            dtype='float32',
            trainable=True
        )

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        euclidean_distance = ops.euclidean_distance(
            tensor.gather('coordinate', 'source'),
            tensor.gather('coordinate', 'target'),
            axis=-1
        )
        return ops.gaussian(
            euclidean_distance, self._loc, self._scale
        )

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'kernels': self.kernels,
        })
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
            if inner_field in ['label', 'weight']:
                if outer_field == 'context':
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


def warn(message: str) -> None:
    warnings.warn(
        message=message,
        category=UserWarning,
        stacklevel=1
    )

def _match_functional_input(functional_input, inputs):
    matching_inputs = {}
    for outer_field, data in functional_input.items():
        matching_inputs[outer_field] = {}
        for inner_field, _ in data.items():
            call_input = inputs[outer_field].pop(inner_field)
            matching_inputs[outer_field][inner_field] = call_input
    unmatching_inputs = inputs
    return matching_inputs, unmatching_inputs

def _add_left_out_inputs(outputs, inputs):
    for outer_field, data in inputs.items():
        for inner_field, value in data.items():
            if inner_field in ['label', 'weight']:
                outputs[outer_field][inner_field] = value
    return outputs 

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
GraphTransformer3D = GTConv3D

