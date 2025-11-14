import keras
import warnings

from molcraft import layers
from molcraft import models
from molcraft import tensors 


@keras.saving.register_keras_serializable(package='molcraft')
class Trainer(models.GraphModel):

    '''Base trainer.

    Wraps and (pre)trains a graph neural network for a certain task.

    Args:
        model: 
            A `models.GraphModel` to be (pre)trained.
    '''
    
    def __init__(self, model: models.GraphModel, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model 
    
    def get_config(self) -> dict:
        config = super().get_config()
        config['model'] = keras.saving.serialize_keras_object(self.model)
        return config
    
    @classmethod
    def from_config(cls, config: dict) -> 'Trainer':
        config['model'] = keras.saving.deserialize_keras_object(config['model'])
        return super().from_config(config)
    

@keras.saving.register_keras_serializable(package='molcraft')
class NodePredictionTrainer(Trainer):

    '''Node prediction trainer.

    Wraps and (pre)trains a graph neural network to perform node predictions.

    Ignores super nodes and edges, if they exist.

    Args:
        model: 
            A `models.GraphModel` to be (pre)trained.
        decoder:
            An optional decoder for converting updated node features to node predictions.
            If None, a `keras.layers.Dense` layer is used with `units` set to `label` dim.
        select_rate:
            The rate of which nodes will be selected for prediction. If None, all nodes are predicted.
        mask_selected:
            Whether to mask the selected nodes. Only relevant if `select_rate` is specified.
        edge_masking_rate:
            The rate of which edges will be masked. If None, edges will not be masked.
            Only relevant if `select_rate` is specified.

    Example:

    >>> import molcraft
    >>> import keras
    >>> 
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer(
    ...     atom_features=[
    ...         molcraft.features.AtomType(['C', 'N', 'O', 'P', 'S']),
    ...     ]
    ... )
    >>> graph = featurizer(['N[C@@H](C)C(=O)O', 'N[C@@H](CS)C(=O)O'])
    >>> # Label nodes with the one-hot encoded atom types for illustration
    >>> graph = graph.update({'node': {'label': graph.node['feature']}})
    >>> 
    >>> inputs = molcraft.layers.Input(graph.spec)
    >>> x = molcraft.layers.NodeEmbedding(128)(inputs)
    >>> x = molcraft.layers.EdgeEmbedding(128)(x)
    >>> x = molcraft.layers.GraphConv(128)(x)
    >>> outputs = molcraft.layers.GraphConv(128)(x)
    >>> model = molcraft.models.GraphModel(inputs, outputs)
    >>> 
    >>> pretrainer = molcraft.trainers.NodePredictionTrainer(
    ...     model,
    ...     decoder=None,       # Dense(units=node_label_dim)
    ...     select_rate=0.5,
    ...     mask_selected=True,
    ... )
    >>> pretrainer.compile(
    ...     optimizer=keras.optimizers.Adam(1e-4),
    ...     loss=keras.losses.CategoricalCrossentropy(from_logits=True),
    ... )
    >>> pretrainer.fit(graph, epochs=10)
    >>> # pretrainer.model.save('/tmp/model.keras')
    '''

    def __init__(
        self,
        model: models.GraphModel,
        decoder: keras.layers.Layer | None = None,
        select_rate: float | None = None,
        mask_selected: bool = False,
        edge_masking_rate: float | None = None,
        **kwargs
    ) -> None:
        super().__init__(model=model, **kwargs)
        
        for layer in self.model.layers:
            if isinstance(layer, layers.NodeEmbedding):
                break
        else:
            raise ValueError('Could not find `NodeEmbedding` layer.')
        
        self._embedder = models.GraphModel(
            self.model.input, layer._symbolic_output
        )
        self._model = models.GraphModel(
            layer._symbolic_output, self.model.output
        )
        self._decoder = decoder
        self._select_rate = select_rate
        self._mask_selected = mask_selected
        if edge_masking_rate and not mask_selected:
            warnings.warn(
                'Setting `edge_masking_rate` to `None`, '
                'as `mask_selected` is set to `False`.'
            )
            edge_masking_rate = None
        self._edge_masking_rate = edge_masking_rate
    
    def build(self, spec: tensors.GraphTensor.Spec) -> None:

        self._has_super = ('super' in spec.node)
        self._has_edge_feature = ('feature' in spec.edge)

        if self._mask_selected:
            node_feature_dim = self._embedder._symbolic_output['node']['feature'].shape[-1]
            self._node_mask_feature = self.get_weight(shape=[node_feature_dim])

        if self._mask_selected and self._has_edge_feature and self._edge_masking_rate:
            edge_feature_dim = self._embedder._symbolic_output['edge']['feature'].shape[-1]
            self._edge_mask_feature = self.get_weight(shape=[edge_feature_dim])
        elif self._edge_masking_rate and not self._has_edge_feature:
            warnings.warn(
                'Setting `edge_masking_rate` to `None`, '
                'as no edge features exist.'
            )
            self._edge_masking_rate = None

        if self._decoder is None:
            label_dim = spec.node['label'].shape[-1]
            self._decoder = keras.layers.Dense(units=label_dim) 

    def propagate(
        self, 
        tensor: tensors.GraphTensor,
        training: bool | None = None,
    ) -> tensors.GraphTensor:
        sample_weight = tensor.node.get('weight')
        if sample_weight is None:
            sample_weight = keras.ops.ones([tensor.num_nodes])

        tensor = self._embedder(tensor)

        if self._select_rate is not None and training:
            # Select nodes to be predicted
            is_not_super = (
                True if not self._has_super else keras.ops.logical_not(tensor.node['super'])
            )
            r = keras.random.uniform(shape=[tensor.num_nodes])
            node_mask = keras.ops.logical_and(is_not_super, self._select_rate > r)
            sample_weight = keras.ops.where(node_mask, sample_weight, 0.0)

            if self._mask_selected:
                # Mask selected node features
                node_feature_masked = keras.ops.where(
                    node_mask[:, None], self._node_mask_feature, tensor.node['feature']
                )
                tensor = tensor.update({'node': {'feature': node_feature_masked}})

                if self._edge_masking_rate:
                    # Mask edge features
                    is_not_super = (
                        True if not self._has_super else keras.ops.logical_not(tensor.edge['super'])
                    )
                    r = keras.random.uniform(shape=[tensor.num_edges])
                    edge_mask = keras.ops.logical_and(is_not_super, self._edge_masking_rate > r)
                    edge_feature_masked = keras.ops.where(
                        edge_mask[:, None], self._edge_mask_feature, tensor.edge['feature']
                    )
                    tensor = tensor.update({'edge': {'feature': edge_feature_masked}})

        node_feature = self._model(tensor).node['feature']
        node_prediction = self._decoder(node_feature)
        return tensor.update({
            'node': {
                'prediction': node_prediction, 
                'sample_weight': sample_weight
            }
        })
    
    def get_config(self) -> dict:
        config = super().get_config()
        config.update({
            'decoder': keras.saving.serialize_keras_object(self._decoder),
            'select_rate': self._select_rate,
            'mask_selected': self._mask_selected,
            'edge_masking_rate': self._edge_masking_rate,
        })
        return config
    
    @classmethod
    def from_config(cls, config: dict) -> 'NodePredictionTrainer':
        config['decoder'] = keras.saving.deserialize_keras_object(config['decoder'])
        return super().from_config(config)