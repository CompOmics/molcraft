import abc
import keras 
import tensorflow as tf
import warnings
from keras.src.models import functional

from molcraft import tensors
from molcraft import ops 


@keras.saving.register_keras_serializable(package='molcraft')
class GraphLayer(keras.layers.Layer):
    """Base graph layer.

    Currently, the `GraphLayer` only supports `GraphTensor` input.

    The list of arguments are only relevant if the derived layer 
    invokes 'get_dense_kwargs`, `get_dense`  or `get_einsum_dense`. 

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
        super().__init__(activity_regularizer=activity_regularizer, **kwargs)
        self._use_bias = use_bias
        self._kernel_initializer = keras.initializers.get(kernel_initializer)
        self._bias_initializer = keras.initializers.get(bias_initializer)
        self._kernel_regularizer = keras.regularizers.get(kernel_regularizer)
        self._bias_regularizer = keras.regularizers.get(bias_regularizer)
        self._kernel_constraint = keras.constraints.get(kernel_constraint)
        self._bias_constraint = keras.constraints.get(bias_constraint)
        self.built = False
        # TODO: Add warning if build is implemented in subclass
        # TODO: Add warning if call is implemented in subclass

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Calls the layer.

        Needs to be implemented by subclass.

        Args:
            tensor:
                A `GraphTensor` instance.
        """
        raise NotImplementedError('`propagate` needs to be implemented.')

    def build_from_spec(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.

        May use built-in methods such as `get_weight`, `get_dense` and `get_einsum_dense`.

        Optionally implemented by subclass. If implemented, it is recommended to
        build the sub-layers via `build([None, input_dim])`. If sub-layers are not 
        built, symbolic input will be passed through the layer to build it.

        Args:
            spec:
                A `GraphTensor.Spec` instance, corresponding to the input `GraphTensor` 
                of the `propagate` method.
        """ 

    def build(self, spec: tensors.GraphTensor.Spec) -> None:
            
        self._custom_build_config = {'spec': _serialize_spec(spec)}

        invoke_build_from_spec = (
            GraphLayer.build_from_spec != self.__class__.build_from_spec
        )
        if invoke_build_from_spec:
            self.build_from_spec(spec)
            self.built = True

        if not self.built:
            # Automatically build layer or model by calling it on symbolic inputs
            self.built = True 
            symbolic_inputs = Input(spec)
            self(symbolic_inputs)
    
    def get_build_config(self) -> dict:
        if not hasattr(self, '_custom_build_config'):
            return super().get_build_config()
        return self._custom_build_config
    
    def build_from_config(self, config: dict) -> None:
        use_custom_build_from_config = ('spec' in config)
        if not use_custom_build_from_config:
            super().build_from_config(config)
        else:
            spec = _deserialize_spec(config['spec'])
            self.build(spec)

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
            activity_regularizer=self.activity_regularizer,
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
            "kernel_constraint": 
                keras.constraints.serialize(self._kernel_constraint),
            "bias_constraint": 
                keras.constraints.serialize(self._bias_constraint),
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class GraphConv(GraphLayer):

    """Base graph neural network layer.
    """
        
    def __init__(self, units: int, **kwargs) -> None:
        super().__init__(**kwargs)
        self.units = units
        
    @abc.abstractmethod
    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Compute messages.

        This method needs to be implemented by subclass.

        Args:
            tensor:
                The inputted `GraphTensor` instance.
        """
    
    @abc.abstractmethod
    def aggregate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Aggregates messages.

        This method needs to be implemented by subclass.

        Args:
            tensor:
                A `GraphTensor` instance containing a message.
        """

    @abc.abstractmethod
    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Updates nodes. 

        This method needs to be implemented by subclass.

        Args:
            tensor:
                A `GraphTensor` instance containing aggregated messages 
                (updated node features).
        """

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Calls the layer.

        The `GraphConv` layer invokes `message`, `aggregate` and `update`
        in sequence.
        
        Args:
            tensor:
                A `GraphTensor` instance.
        """
        tensor = self.message(tensor)
        tensor = self.aggregate(tensor)
        tensor = self.update(tensor)
        return tensor 

    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'units': self.units
        })
        return config
    
    
@keras.saving.register_keras_serializable(package='molcraft')
class Projection(GraphLayer):
    """Base graph projection layer.
    """
    def __init__(
        self, 
        units: int = None, 
        activation: str = None, 
        field: str = 'node',
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.units = units
        self._activation = keras.activations.get(activation)
        self.field = field 

    def build_from_spec(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        data = getattr(spec, self.field, None)
        if data is None:
            raise ValueError('Could not access field {self.field!r}.')
        feature_dim = data['feature'].shape[-1]
        if not self.units:
            self.units = feature_dim
        self._dense = self.get_dense(self.units)
        self._dense.build([None, feature_dim])

    def propagate(self, tensor: tensors.GraphTensor):
        """Calls the layer.
        """
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
class GraphNetwork(GraphLayer):

    """Graph neural network.

    Sequentially calls graph layers (`GraphLayer`) and concatenates its output. 

    Args:
        layers (list):
            A list of graph layers.
    """

    def __init__(self, layers: list[GraphLayer], **kwargs) -> None:
        super().__init__(**kwargs)
        self.layers = layers
        self._update_edge_feature = False

    def build_from_spec(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        units = self.layers[0].units 
        node_feature_dim = spec.node['feature'].shape[-1]
        if node_feature_dim != units:
            warn(
                'Node feature dim does not match `units` of the first layer. '
                'Automatically adding a node projection layer to match `units`.'
            )
            self._node_dense = self.get_dense(units)
            self._update_node_feature = True 
        has_edge_feature = 'feature' in spec.edge 
        if has_edge_feature:
            edge_feature_dim = spec.edge['feature'].shape[-1]
            if edge_feature_dim != units:
                warn(
                    'Edge feature dim does not match `units` of the first layer. '
                    'Automatically adding a edge projection layer to match `units`.'
                )
                self._edge_dense = self.get_dense(units)
                self._update_edge_feature = True

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Calls the layer.
        """
        x = tensors.to_dict(tensor)
        if self._update_node_feature:
            x['node']['feature'] = self._node_dense(tensor.node['feature'])
        if self._update_edge_feature:
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

        Args:
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
        embed_context: bool = True,
        allow_masking: bool = True, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self._embed_context = embed_context
        self._masking_rate = None
        self._allow_masking = allow_masking

    def build_from_spec(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        feature_dim = spec.node['feature'].shape[-1]
        if not self.dim:
            self.dim = feature_dim
        self._node_dense = self.get_dense(self.dim)
        self._node_dense.build([None, feature_dim])

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
            self._context_dense.build([None, context_feature_dim])

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Calls the layer.
        """
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

        return tensor.update({'node': {'feature': feature}})

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
            'allow_masking': self._allow_masking
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
        allow_masking: bool = True, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.dim = dim
        self._masking_rate = None
        self._allow_masking = allow_masking

    def build_from_spec(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        feature_dim = spec.edge['feature'].shape[-1]
        if not self.dim:
            self.dim = feature_dim
        self._edge_dense = self.get_dense(self.dim)
        self._edge_dense.build([None, feature_dim])

        self._has_super = 'super' in spec.edge
        if self._has_super:
            self._super_feature = self.get_weight(shape=[self.dim], name='super_edge_feature')
        if self._allow_masking:
            self._mask_feature = self.get_weight(shape=[self.dim], name='mask_edge_feature')

    def propagate(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Calls the layer.
        """
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

        return tensor.update({'edge': {'feature': feature}})

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
            'allow_masking': self._allow_masking
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class ContextProjection(Projection):
    """Context projection layer.
    """
    def __init__(self, units: int = None, activation: str = None, **kwargs):
        super().__init__(units=units, activation=activation, field='context', **kwargs)


@keras.saving.register_keras_serializable(package='molcraft')
class NodeProjection(Projection):
    """Node projection layer.
    """
    def __init__(self, units: int = None, activation: str = None, **kwargs):
        super().__init__(units=units, activation=activation, field='node', **kwargs)


@keras.saving.register_keras_serializable(package='molcraft')
class EdgeProjection(Projection):
    """Edge projection layer.
    """
    def __init__(self, units: int = None, activation: str = None, **kwargs):
        super().__init__(units=units, activation=activation, field='edge', **kwargs)


@keras.saving.register_keras_serializable(package='molcraft')
class GINConv(GraphConv):

    """Graph isomorphism network layer.
    """

    def __init__(
        self,
        units: int,
        activation: keras.layers.Activation | str | None = 'relu',
        dropout: float = 0.0,
        normalize: bool = True,
        update_edge_feature: bool = True,
        **kwargs,
    ):
        super().__init__(units=units, **kwargs)
        self._activation = keras.activations.get(activation)
        self._normalize = normalize
        self._dropout = dropout 
        self._update_edge_feature = update_edge_feature

    def build_from_spec(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        node_feature_dim = spec.node['feature'].shape[-1]

        self.epsilon = self.add_weight(
            name='epsilon', 
            shape=(), 
            initializer='zeros',
            trainable=True,
        )

        if 'feature' in spec.edge:
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
                self._edge_dense.build([None, edge_feature_dim])
        else:
            self._update_edge_feature = False
                
        has_overridden_update = self.__class__.update != GINConv.update 
        if not has_overridden_update:
            # Use default feedforward network
            self._feedforward_intermediate_dense = self.get_dense(self.units)
            self._feedforward_intermediate_dense.build([None, node_feature_dim])

            if self._normalize:
                self._feedforward_intermediate_norm = keras.layers.BatchNormalization()
                self._feedforward_intermediate_norm.build([None, self.units])

            self._feedforward_dropout = keras.layers.Dropout(self._dropout)
            self._feedforward_activation = self._activation
                
            self._feedforward_output_dense = self.get_dense(self.units)
            self._feedforward_output_dense.build([None, self.units])
    
    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Compute messages.
        """
        message = tensor.gather('feature', 'source')
        edge_feature = tensor.edge.get('feature')
        if self._update_edge_feature:
            edge_feature = self._edge_dense(edge_feature)
        if edge_feature is not None:
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
        """Aggregates messages.
        """
        node_feature = tensor.aggregate('message')
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
        """Updates nodes. 
        """
        node_feature = tensor.node['feature']
        node_feature = self._feedforward_intermediate_dense(node_feature)
        node_feature = self._feedforward_activation(node_feature)
        if self._normalize:
            node_feature = self._feedforward_intermediate_norm(node_feature)
        node_feature = self._feedforward_dropout(node_feature)
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
            'activation': keras.activations.serialize(self._activation),
            'dropout': self._dropout, 
            'normalize': self._normalize,
        })
        return config


@keras.saving.register_keras_serializable(package='molcraft')
class GTConv(GraphConv):

    """Graph transformer layer.
    """

    def __init__(
        self,
        units: int,
        heads: int = 8,
        activation: keras.layers.Activation | str | None = "relu",
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        normalize: bool = True,
        normalize_first: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(units=units, **kwargs)
        self._heads = heads
        if self.units % self.heads != 0:
            raise ValueError(f"units need to be divisible by heads.")
        self._head_units = self.units // self.heads 
        self._activation = keras.activations.get(activation)
        self._dropout = dropout
        self._attention_dropout = attention_dropout
        self._normalize = normalize
        self._normalize_first = normalize_first

    @property 
    def heads(self):
        return self._heads 
    
    @property 
    def head_units(self):
        return self._head_units 
    
    def build_from_spec(self, spec):
        """Builds the layer.
        """
        node_feature_dim = spec.node['feature'].shape[-1]
        incompatible_dim = node_feature_dim != self.units
        if incompatible_dim:
            warnings.warn(
                message=(
                    '`GTConv` uses residual connections, but input node feature dim '
                    'is incompatible with intermediate dim (`units`). '
                    'Automatically projecting first residual to match its dim with intermediate dim.'
                ),
                category=UserWarning,
                stacklevel=1
            )
            self._residual_dense = self.get_dense(self.units)
            self._residual_dense.build([None, node_feature_dim])
            self._project_residual = True
        else:
            self._project_residual = False
            
        self._query_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._query_dense.build([None, node_feature_dim])

        self._key_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._key_dense.build([None, node_feature_dim])

        self._value_dense = self.get_einsum_dense(
            'ij,jkh->ikh', (self.head_units, self.heads)
        )
        self._value_dense.build([None, node_feature_dim])

        self._output_dense = self.get_dense(self.units)
        self._output_dense.build([None, self.units])

        self._softmax_dropout = keras.layers.Dropout(self._attention_dropout) 

        self._self_attention_norm = keras.layers.LayerNormalization()
        if self._normalize_first:
            self._self_attention_norm.build([None, node_feature_dim])
        else:
            self._self_attention_norm.build([None, self.units])

        self._self_attention_dropout = keras.layers.Dropout(self._dropout)

        has_overriden_edge_bias = (
            self.__class__.add_edge_bias != GTConv.add_edge_bias
        )
        if not has_overriden_edge_bias:
            self._has_edge_length = 'length' in spec.edge
            if self._has_edge_length and 'bias' not in spec.edge:
                edge_length_dim = spec.edge['length'].shape[-1]
                self._spatial_encoding_dense = self.get_einsum_dense(
                    'ij,jkh->ikh', (1, self.heads), kernel_initializer='zeros'
                )
                self._spatial_encoding_dense.build([None, edge_length_dim])

            self._has_edge_feature = 'feature' in spec.edge
            if self._has_edge_feature and 'bias' not in spec.edge:
                edge_feature_dim = spec.edge['feature'].shape[-1]
                self._edge_feature_dense = self.get_einsum_dense(
                    'ij,jkh->ikh', (1, self.heads),
                )
                self._edge_feature_dense.build([None, edge_feature_dim])

        has_overridden_update = self.__class__.update != GTConv.update 
        if not has_overridden_update:

            self._feedforward_norm = keras.layers.LayerNormalization()
            self._feedforward_norm.build([None, self.units])

            self._feedforward_dropout = keras.layers.Dropout(self._dropout)

            self._feedforward_intermediate_dense = self.get_dense(self.units)
            self._feedforward_intermediate_dense.build([None, self.units])

            self._feedforward_output_dense = self.get_dense(self.units)
            self._feedforward_output_dense.build([None, self.units])

    def add_node_bias(self, tensor: tensors.GraphTensor) -> tf.Tensor:
        return tensor 
    
    def add_edge_bias(self, tensor: tensors.GraphTensor) -> tf.Tensor:
        if 'bias' in tensor.edge:
            return tensor 
        elif not self._has_edge_feature and not self._has_edge_length:
            return tensor 
        
        if self._has_edge_feature and not self._has_edge_length:
            edge_bias = self._edge_feature_dense(tensor.edge['feature'])
        elif not self._has_edge_feature and self._has_edge_length:
            edge_bias = self._spatial_encoding_dense(tensor.edge['length'])
        else:
            edge_bias = (
                self._edge_feature_dense(tensor.edge['feature']) + 
                self._spatial_encoding_dense(tensor.edge['length'])
            )
 
        return tensor.update(
            {
                'edge': {
                    'bias': edge_bias
                }
            }
        )

    def message(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Compute messages.
        """
        tensor = self.add_edge_bias(tensor)
        tensor = self.add_node_bias(tensor)

        node_feature = tensor.node['feature']

        if 'bias' in tensor.node:
            node_feature += tensor.node['bias']
        
        if self._normalize_first:
            node_feature = self._self_attention_norm(node_feature)
        
        query = self._query_dense(node_feature)
        key = self._key_dense(node_feature)
        value = self._value_dense(node_feature)

        query = ops.gather(query, tensor.edge['source'])
        key = ops.gather(key, tensor.edge['target'])
        value = ops.gather(value, tensor.edge['source'])

        attention_score = keras.ops.sum(query * key, axis=1, keepdims=True)
        attention_score /= keras.ops.sqrt(float(self.units))

        if 'bias' in tensor.edge:
            attention_score += tensor.edge['bias']
        
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
        """Aggregates messages.
        """
        node_feature = tensor.aggregate('message')

        node_feature = keras.ops.reshape(node_feature, (-1, self.units))
        node_feature = self._output_dense(node_feature)
        node_feature = self._self_attention_dropout(node_feature)

        residual = tensor.node['feature']
        if self._project_residual:
            residual = self._residual_dense(residual)
        node_feature += residual

        if not self._normalize_first:
            node_feature = self._self_attention_norm(node_feature)

        return tensor.update(
            {
                'node': {
                    'feature': node_feature,
                },
                'edge': {
                    'message': None,
                    'weight': None,
                }
            }
        )
    

    def update(self, tensor: tensors.GraphTensor) -> tensors.GraphTensor:
        """Updates nodes. 
        """
        node_feature = tensor.node['feature']

        if self._normalize_first:
            node_feature = self._feedforward_norm(node_feature)

        node_feature = self._feedforward_intermediate_dense(node_feature)
        node_feature = self._activation(node_feature)
        node_feature = self._feedforward_output_dense(node_feature)

        node_feature = self._feedforward_dropout(node_feature)
        node_feature += tensor.node['feature']

        if not self._normalize_first:
            node_feature = self._feedforward_norm(node_feature)

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
            'activation': keras.activations.serialize(self._activation),
            'dropout': self._dropout, 
            'attention_dropout': self._attention_dropout,
            'normalize': self._normalize,
            'normalize_first': self._normalize_first,
        })
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class Readout(keras.layers.Layer):

    def __init__(self, mode: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.mode = mode
        if not self.mode:
            self._reduce_fn = None
        elif str(self.mode).lower().startswith('sum'):
            self._reduce_fn = keras.ops.segment_sum
        elif str(self.mode).lower().startswith('max'):
            self._reduce_fn = keras.ops.segment_max 
        elif str(self.mode).lower().startswith('super'):
            self._reduce_fn = keras.ops.segment_sum
        else:
            self._reduce_fn = ops.segment_mean

    def build_from_spec(self, spec: tensors.GraphTensor.Spec) -> None:
        """Builds the layer.
        """
        pass

    def reduce(self, tensor: tensors.GraphTensor) -> tf.Tensor:
        if self._reduce_fn is None:
            raise NotImplementedError("Need to define a reduce method.")
        if str(self.mode).lower().startswith('super'):
            node_feature = keras.ops.where(
                tensor.node['super'][:, None], tensor.node['feature'], 0.0
            )
            return self._reduce_fn(
                node_feature, tensor.graph_indicator, tensor.num_subgraphs
            )
        return self._reduce_fn(
            tensor.node['feature'], tensor.graph_indicator, tensor.num_subgraphs
        )

    def build(self, input_shapes) -> None:
        spec = tensors.GraphTensor.Spec.from_input_shape_dict(input_shapes)
        self.build_from_spec(spec)
        self.built = True 

    def call(self, graph) -> tf.Tensor:
        graph_tensor = tensors.from_dict(graph) 
        if tensors.is_ragged(graph_tensor):
            graph_tensor = graph_tensor.flatten()
        return self.reduce(graph_tensor)

    def __call__(
        self, 
        graph: tensors.GraphTensor, 
        *args, 
        **kwargs
    ) -> tensors.GraphTensor:
        is_tensor = isinstance(graph, tensors.GraphTensor)
        if is_tensor:
            graph = tensors.to_dict(graph)
        tensor = super().__call__(graph, *args, **kwargs)
        return tensor
    
    def get_config(self) -> dict:
        config = super().get_config()
        config['mode'] = self.mode 
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


GraphTransformer = GTConvolution = GTConv
GINConvolution = GINConv

EdgeEmbed = EdgeEmbedding
NodeEmbed = NodeEmbedding

ContextDense = ContextProjection
EdgeDense = EdgeProjection
NodeDense = NodeProjection

