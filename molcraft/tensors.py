import tensorflow as tf 
import keras
import typing 
from tensorflow.python.framework import composite_tensor

from molcraft import ops


class GraphTensorBatchEncoder(tf.experimental.ExtensionTypeBatchEncoder):
    
    def batch(self, spec: 'GraphTensor.Spec', batch_size: int | None):
        """Batches spec.
        """

        def batch_field(f):
            if isinstance(f, tf.TensorSpec):
                return tf.TensorSpec(
                    shape=[None] + f.shape[1:],
                    dtype=f.dtype)
            elif isinstance(f, tf.RaggedTensorSpec):
                return tf.RaggedTensorSpec(
                    shape=[batch_size, None] + f.shape[1:],
                    dtype=f.dtype,
                    ragged_rank=1,
                    row_splits_dtype=f.row_splits_dtype)
            elif isinstance(f, tf.TypeSpec):
                return f.__batch_encoder__.batch(f, batch_size)
            return f
        fields = dict(spec.__dict__)
        # Pop context fields as they will be batched differently
        context_fields = fields.pop('context')
        batched_fields = tf.nest.map_structure(batch_field, fields)
        batched_spec = object.__new__(type(spec))
        batched_context_fields = tf.nest.map_structure(
            lambda spec: tf.TensorSpec([batch_size] + spec.shape, spec.dtype), 
            context_fields)
        batched_spec.__dict__.update({'context': batched_context_fields})
        batched_spec.__dict__.update(batched_fields)
        return batched_spec
    
    def unbatch(self, spec: 'GraphTensor.Spec'):
        """Unbatches spec.
        """

        def unbatch_field(f):   
            if isinstance(f, tf.TensorSpec):
                return tf.TensorSpec(
                    shape=[None] + f.shape[1:],
                    dtype=f.dtype)
            elif isinstance(f, tf.RaggedTensorSpec):
                return tf.RaggedTensorSpec(
                    shape=[None] + f.shape[2:],
                    dtype=f.dtype,
                    ragged_rank=0,
                    row_splits_dtype=f.row_splits_dtype)
            elif isinstance(f, tf.TypeSpec):
                return f.__batch_encoder__.unbatch(f)
            return f
        fields = dict(spec.__dict__)
        # Pop context fields as they will be unbatched differently
        context_fields = fields.pop('context')
        unbatched_fields = tf.nest.map_structure(unbatch_field, fields)
        unbatched_context_fields = tf.nest.map_structure(
            lambda spec: tf.TensorSpec(spec.shape[1:], spec.dtype), 
            context_fields)
        unbatched_spec = object.__new__(type(spec))
        unbatched_spec.__dict__.update({'context': unbatched_context_fields})
        unbatched_spec.__dict__.update(unbatched_fields)
        return unbatched_spec
        
    def encode(self, spec: 'GraphTensor.Spec', value: 'GraphTensor', minimum_rank: int = 0):
        """Encodes value.
        """
        unflatten = False if (is_ragged(spec) or is_scalar(spec)) else True 
        if unflatten:
            value = value.unflatten()
        value_components = tuple(value.__dict__[key] for key in spec.__dict__)
        value_components = tuple(
            x for x in tf.nest.flatten(value_components) 
            if isinstance(x, (tf.Tensor, composite_tensor.CompositeTensor))
        )
        return value_components
    
    def encoding_specs(self, spec: 'GraphTensor.Spec'):
        """Matches spec and encoded value of `encode(spec, value)`.
        """
        def encode_fields(f):
            if isinstance(f, tf.TensorSpec):
                scalar = is_scalar(spec)
                return tf.RaggedTensorSpec(
                    shape=([None] if scalar else [None, None]) + f.shape[1:], 
                    dtype=f.dtype, 
                    ragged_rank=(0 if scalar else 1),
                    row_splits_dtype=spec.context['size'].dtype)
            return f
        fields = dict(spec.__dict__)
        context_fields = fields.pop('context')
        encoded_fields = tf.nest.map_structure(encode_fields, fields)
        encoded_fields = {**{'context': context_fields}, **encoded_fields}
        spec_components = tuple(encoded_fields.values())
        spec_components = tuple(
            x for x in tf.nest.flatten(spec_components) 
            if isinstance(x, tf.TypeSpec)
        )
        return spec_components
    
    def decode(self, spec, encoded_value):
        """Decodes encoded value.
        """
        spec_tuple = tuple(spec.__dict__.values())
        encoded_value = iter(encoded_value)
        value_tuple = [
            next(encoded_value) if isinstance(x, tf.TypeSpec) else x
            for x in tf.nest.flatten(spec_tuple)
        ]
        value_tuple = tf.nest.pack_sequence_as(spec_tuple, value_tuple)
        fields = dict(zip(spec.__dict__.keys(), value_tuple))
        value = object.__new__(spec.value_type)
        value.__dict__.update(fields)

        flatten = is_ragged(value) and not is_ragged(spec)
        if flatten:
            value = value.flatten()
        return value
    

class GraphTensor(tf.experimental.BatchableExtensionType):
    context: typing.Mapping[str, typing.Union[tf.Tensor, tf.RaggedTensor]]
    node: typing.Mapping[str, typing.Union[tf.Tensor, tf.RaggedTensor]]
    edge: typing.Mapping[str, typing.Union[tf.Tensor, tf.RaggedTensor]]

    __batch_encoder__ = GraphTensorBatchEncoder()

    __name__ = 'GraphTensor'

    def __validate__(self):
        if tf.executing_eagerly():
            assert 'size' in self.context, "graph.context['size'] is required."
            assert self.context['size'].dtype == tf.int32, (
                "dtype of graph.context['size'] needs to be int32."
            )
            assert 'feature' in self.node, "graph.node['feature'] is required."
            assert 'source' in self.edge, "graph.edge['source'] is required."
            assert 'target' in self.edge, "graph.edge['target'] is required."
            assert self.edge['source'].dtype == tf.int32, (
                "dtype of graph.edge['source'] needs to be int32."
            )
            assert self.edge['target'].dtype == tf.int32, (
                "dtype of graph.edge['target'] needs to be int32."
            )
            if isinstance(self.node['feature'], tf.Tensor):
                num_nodes = keras.ops.shape(self.node['feature'])[0]
            else:
                num_nodes = keras.ops.sum(self.node['feature'].row_lengths())
            assert keras.ops.sum(self.context['size']) == num_nodes, (
                "graph.node['feature'] tensor is incompatible with graph.context['size']"
            )
        
    @property 
    def spec(self):
        def unspecify_spec(s):
            if isinstance(s, tf.TensorSpec):
                return tf.TensorSpec([None] + s.shape[1:], s.dtype)
            return s
        orig_spec = tf.type_spec_from_value(self)
        fields = dict(orig_spec.__dict__)
        context_fields = fields.pop('context')
        new_spec_components = tf.nest.map_structure(unspecify_spec, fields)
        new_spec_components['context'] = context_fields
        return orig_spec.__class__(**new_spec_components)
    
    @property 
    def shape(self):
        if is_ragged(self):
            return self.node['feature'].shape 
        return self.context['size'].shape + [None] + self.node['feature'].shape[1:] 
    
    @property 
    def dtype(self):
        return self.node['feature'].dtype

    @property
    def graph_indicator(self):
        dtype = self.context['size'].dtype
        if is_scalar(self):
            return tf.zeros(tf.shape(self.node['feature'])[:1], dtype=dtype)
        num_graphs = keras.ops.shape(self.context['size'])[0]
        if is_ragged(self):
            return keras.ops.arange(num_graphs, dtype=dtype)
        return keras.ops.repeat(keras.ops.arange(num_graphs, dtype=dtype), self.context['size'])

    @property
    def num_subgraphs(self) -> tf.Tensor:
        dtype = self.context['size'].dtype
        if is_scalar(self):
            num_subgraphs = tf.constant(1, dtype=dtype)
        else:
            num_subgraphs = tf.shape(self.context['size'], out_type=dtype)[0]
        if tf.executing_eagerly():
            return int(num_subgraphs)
        return num_subgraphs
    
    @property 
    def num_nodes(self):
        num_nodes = keras.ops.shape(self.node['feature'])[0]
        if tf.executing_eagerly():
            return int(num_nodes)
        return num_nodes 
    
    @property 
    def num_edges(self):
        num_edges = keras.ops.shape(self.edge['source'])[0]
        if tf.executing_eagerly():
            return int(num_edges)
        return num_edges 
    
    def gather(self, node_attr: str, edge_type: str) -> tf.Tensor:
        if edge_type != 'source' and edge_type != 'target':
            raise ValueError
        return ops.gather(self.node[node_attr], self.edge[edge_type])
    
    def aggregate(self, edge_attr: str, edge_type: str = 'target', mode: str = 'sum') -> tf.Tensor:
        if edge_type != 'source' and edge_type != 'target':
            raise ValueError('`edge_attr` needs to be `source` or `target`.')
        edge_attr = self.edge[edge_attr]
        if 'weight' in self.edge:
            edge_attr = ops.edge_weight(edge_attr, self.edge['weight'])
        return ops.aggregate(edge_attr, self.edge[edge_type], self.num_nodes, mode=mode)
    
    def propagate(self, add_edge_feature: bool = False):
        updated_feature = ops.propagate(
            node_feature=self.node['feature'],
            edge_source=self.edge['source'],
            edge_target=self.edge['target'],
            edge_feature=self.edge.get('feature') if add_edge_feature else None,
            edge_weight=self.edge.get('weight'),
        )
        return self.update({'node': {'feature': updated_feature}})

    def flatten(self):
        if not is_ragged(self):
            raise ValueError(
                f"{self.__class__.__qualname__} instance is already flat.")
        def flatten_fn(x):
            if isinstance(x, tf.RaggedTensor):
                return x.flat_values
            return x
        edge_increment = ops.gather(
            self.node['feature'].row_starts(), self.edge['source'].value_rowids())
        edge_increment = tf.cast(
            edge_increment, dtype=self.edge['source'].dtype)
        data = to_dict(self)
        flat_values = tf.nest.map_structure(flatten_fn, data)
        flat_values['edge']['source'] += edge_increment
        flat_values['edge']['target'] += edge_increment 
        return from_dict(flat_values)
        
    def unflatten(self, *, force: bool = False):
        if is_scalar(self):
            raise ValueError(
                f"{self.__class__.__qualname__} instance is a scalar " 
                "and cannot be unflattened.")
        if is_ragged(self):
            raise ValueError(
                f"{self.__class__.__qualname__} instance is already unflat.")
        def unflatten_fn(x, value_rowids, nrows) -> tf.RaggedTensor:
            if isinstance(x, tf.Tensor):
                return tf.RaggedTensor.from_value_rowids(x, value_rowids, nrows)
            return x
        graph_indicator_node = self.graph_indicator
        graph_indicator_edge = ops.gather(graph_indicator_node, self.edge['source'])
        if force:
            sorted_indices = keras.ops.argsort(graph_indicator_edge)
        num_subgraphs = self.num_subgraphs
        unflat_values = {}
        data = to_dict(self)
        for key in tf.type_spec_from_value(self).__dict__:
            value = data[key]
            if key == 'context':
                unflat_values[key] = value
            elif key == 'node':
                unflat_values[key] = tf.nest.map_structure(
                    lambda x: unflatten_fn(x, graph_indicator_node, num_subgraphs),
                    value)
                row_starts = unflat_values[key]['feature'].row_starts()
                edge_decrement = ops.gather(row_starts, graph_indicator_edge)
                if force:
                    edge_decrement = ops.gather(edge_decrement, sorted_indices)
                    graph_indicator_edge = ops.gather(graph_indicator_edge, sorted_indices)
                edge_decrement = tf.cast(edge_decrement, dtype=self.edge['source'].dtype)
            elif key == 'edge':
                if force:
                    value = tf.nest.map_structure(lambda x: ops.gather(x, sorted_indices), value)
                value['source'] -= edge_decrement
                value['target'] -= edge_decrement
                unflat_values[key] = tf.nest.map_structure(
                    lambda x: unflatten_fn(x, graph_indicator_edge, num_subgraphs),
                    value)
        return from_dict(unflat_values)

    def update(self, values):
        data = to_dict(self)
        for outer_field, mapping in values.items():
            if outer_field == 'context':
                reference_value = data[outer_field]['size']
            elif outer_field == 'edge':
                reference_value = data[outer_field]['source']
            else:
                reference_value = data[outer_field]['feature']
            for inner_field, value in mapping.items():
                if value is None:
                    data[outer_field].pop(inner_field, None)
                    continue
                data[outer_field][inner_field] = _maybe_convert_new_value(
                    value, reference_value
                )
        return self.__class__(**data)

    def replace(self, values):
        data = to_dict(self)
        for outer_field, mapping in values.items():
            if outer_field == 'context':
                reference_value = data[outer_field]['size']
            elif outer_field == 'edge':
                reference_value = data[outer_field]['source']
            else:
                reference_value = data[outer_field]['feature']
            for inner_field, value in mapping.items():
                values[outer_field][inner_field] = _maybe_convert_new_value(
                    value, reference_value
                )
        return self.__class__(**values)
    
    def __getitem__(self, index):
        if index is None and is_scalar(self):
            return self.__class__(
                context={key: value[None] for (key, value) in self.context.items()},
                node=self.node,
                edge=self.edge,
            )
        if not is_ragged(self):
            is_flat = True
            tensor = self.unflatten()
        else:
            tensor = self
            is_flat = False
        data = to_dict(tensor)
        if isinstance(index, (slice, int)):
            data = tf.nest.map_structure(lambda x: x[index], data)
        else:
            data = tf.nest.map_structure(lambda x: ops.gather(x, index), data)
        tensor = from_dict(data)
        if is_flat and not is_scalar(tensor):
            return tensor.flatten()
        return tensor
    
    def __repr__(self):
        return _repr(self)
    
    def numpy(self):
        """For now added to work with `keras.Model.predict`"""
        return self 

    class Spec:

        def __init__(
            self, 
            context: typing.Mapping[str, tf.TensorSpec | tf.RaggedTensorSpec], 
            node: typing.Mapping[str, tf.TensorSpec | tf.RaggedTensorSpec],  
            edge: typing.Mapping[str, tf.TensorSpec | tf.RaggedTensorSpec],  
        ) -> None:
            self.context = context 
            self.node = node
            self.edge = edge 

        @property
        def shape(self):
            if is_ragged(self):
                return self.node['feature'].shape 
            return self.context['size'].shape + [None] + self.node['feature'].shape[1:] 

        @classmethod
        def from_input_shape_dict(cls, input_shape: dict[str, tf.TensorShape]) -> 'GraphTensor.Spec':
            for key, value in input_shape.items():
                input_shape[key] = {k: tf.TensorShape(v) for k, v in value.items()}
            return cls(**tf.nest.map_structure(lambda s: tf.TensorSpec(s, dtype=tf.variant), input_shape)) 

        def __repr__(self):
            return _repr(self)
    
    
@tf.experimental.dispatch_for_api(tf.concat, {'values': typing.List[GraphTensor]})
def graph_tensor_concat(
    values: typing.List[GraphTensor],
    axis: int = 0,
    name: str = 'concat'
) -> GraphTensor:
    ragged = [is_ragged(v) for v in values]
    if 0 < sum(ragged) < len(ragged):
        raise ValueError(
            'Nested data of the GraphTensor instances do not have consistent '
            'types: found both tf.RaggedTensor values and tf.Tensor values.')
    else:
        ragged = ragged[0]

    if ragged:
        values = [v.flatten() for v in values]

    flat_values = [tf.nest.flatten(v, expand_composites=True) for v in values]
    flat_values = [tf.concat(f, axis=0) for f in list(zip(*flat_values))]
    num_edges = [keras.ops.shape(v.edge['source'])[0] for v in values]
    num_nodes = [keras.ops.shape(v.node['feature'])[0] for v in values]
    incr = tf.concat([[0], tf.cumsum(num_nodes)[:-1]], axis=0)
    incr = tf.repeat(incr, num_edges)
    value = tf.nest.pack_sequence_as(values[0], flat_values, expand_composites=True)

    edge_update = {
        'source': value.edge['source'] + incr,
        'target': value.edge['target'] + incr,
    }
    value = value.update(
        {
            'edge': edge_update
        },
    )
    if not ragged:
        return value
    return value.unflatten()

# TODO: Clean this up.
@tf.experimental.dispatch_for_api(tf.stack, {'values': typing.List[GraphTensor]})
def graph_tensor_stack(
    values: typing.List[GraphTensor],
    axis: int = 0,
    name: str = 'stack'
) -> GraphTensor:
    ragged = [is_ragged(v) for v in values]
    if not is_scalar(values[0]):
        raise ValueError(
            'tf.stack on a list of `GraphTensor`s is currently '
            'only supported for scalar `GraphTensor`s. '
        )
    if any(ragged):
        raise ValueError(
            'tf.stack on a list of `GraphTensor`s is currently '
            'only supported for flattened `GraphTensor`s. '
        )
    
    def concat_or_stack(k, v):
        if k.startswith('context'):
            return tf.stack(v, axis=0)
        return tf.concat(v, axis=0)

    fields = tuple(tf.type_spec_from_value(values[0]).__dict__)
    num_inner_fields = tuple(len(values[0].__dict__[field]) for field in fields)
    outer_keys = []
    for (f, num_fields) in zip(fields, num_inner_fields):
        outer_keys.extend([f] * num_fields)

    flat_values = [tf.nest.flatten(v, expand_composites=True) for v in values]
    flat_values = [concat_or_stack(k, f) for k, f in zip(outer_keys, list(zip(*flat_values)))]
    value = tf.nest.pack_sequence_as(values[0], flat_values, expand_composites=True)

    num_edges = [keras.ops.shape(v.edge['source'])[0] for v in values]
    num_nodes = [keras.ops.shape(v.node['feature'])[0] for v in values]
    incr = tf.concat([[0], tf.cumsum(num_nodes)[:-1]], axis=0)
    incr = tf.repeat(incr, num_edges)
    edge_update = {
        'source': value.edge['source'] + incr,
        'target': value.edge['target'] + incr,
    }
    value = value.update(
        {
            'edge': edge_update
        },
    )
    return value

def is_scalar(value_or_spec: GraphTensor | GraphTensor.Spec) -> bool:
    return value_or_spec.context['size'].shape.rank == 0

def is_ragged(value_or_spec: GraphTensor | GraphTensor.Spec) -> bool:
    is_ragged = isinstance(
        value_or_spec.node['feature'], (tf.RaggedTensor, tf.RaggedTensorSpec))
    if isinstance(value_or_spec, tf.RaggedTensorSpec):
        is_ragged = (
            is_ragged and value_or_spec.node['feature'].ragged_rank == 1)
    return is_ragged

def to_dict(tensor: GraphTensor) -> dict:
    spec = tf.type_spec_from_value(tensor)
    return {key: dict(tensor.__dict__[key]) for key in spec.__dict__}

def from_dict(data: dict) -> GraphTensor:
    data['context']['size'] = tf.cast(data['context']['size'], tf.int32)
    data['edge']['source'] = tf.cast(data['edge']['source'], tf.int32)
    data['edge']['target'] = tf.cast(data['edge']['target'], tf.int32)
    return GraphTensor(**data)

def is_graph(data):
    if isinstance(data, GraphTensor):
        return True 
    elif isinstance(data, dict) and 'size' in data.get('context', {}):
        return True 
    return False
    
def _maybe_convert_new_value(
    new_value: tf.Tensor | tf.RaggedTensor, 
    old_value: tf.Tensor | tf.RaggedTensor | None,
) -> tf.Tensor | tf.RaggedTensor:
    if old_value is None:
        return new_value
    is_old_ragged = isinstance(old_value, tf.RaggedTensor)
    is_new_ragged = isinstance(new_value, tf.RaggedTensor)
    if is_old_ragged and not is_new_ragged:
        new_value = old_value.with_flat_values(new_value)
    elif not is_old_ragged and is_new_ragged:
        new_value = new_value.flat_values
    return new_value

def _repr(x: GraphTensor | GraphTensor.Spec):
    if isinstance(x, GraphTensor):
        def _trepr(v: tf.Tensor | tf.RaggedTensor):
            if isinstance(v, tf.Tensor):
                return f'<tf.Tensor: shape={v.shape.as_list()}, dtype={v.dtype.name}>'
            return (
                f'<tf.RaggedTensor: shape={v.shape.as_list()}, ' 
                f'dtype={v.dtype.name}, ragged_rank={v.ragged_rank}>'
            )
    else:
        def _trepr(v: tf.TensorSpec | tf.RaggedTensorSpec):
            if isinstance(v, tf.TensorSpec):
                return f'<tf.TensorSpec: shape={v.shape.as_list()}, dtype={v.dtype.name}>'
            return (
                f'<tf.RaggedTensorSpec: shape={v.shape.as_list()}, ' 
                f'dtype={v.dtype.name}, ragged_rank={v.ragged_rank}>'
            )
        
    context_fields = f',\n        '.join([f'{k!r}: {_trepr(v)}' for k, v in x.context.items()])
    node_fields = f',\n        '.join([f'{k!r}: {_trepr(v)}'for k, v in x.node.items()])
    edge_fields = f',\n        '.join([f'{k!r}: {_trepr(v)}' for k, v in x.edge.items()])

    context_field = 'context={\n        ' + context_fields + '\n    }' 
    node_field = 'node={\n        ' + node_fields + '\n    }' 
    edge_field = 'edge={\n        ' + edge_fields + '\n    }' 

    fields = ',\n    '.join([context_field, node_field, edge_field])
    return x.__class__.__name__ + '(\n    ' + fields + '\n)'