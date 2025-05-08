import keras 
import json
import abc
import typing 
import copy 
import warnings
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp

from pathlib import Path

from molcraft import tensors 
from molcraft import features
from molcraft import chem
from molcraft import conformers
from molcraft import descriptors


@keras.saving.register_keras_serializable(package='molcraft')
class Featurizer(abc.ABC):

    """Base class for featurizers.
    """

    @abc.abstractmethod
    def call(
        self, 
        x: tensors.GraphTensor
    ) -> tensors.GraphTensor | list[tensors.GraphTensor]:
        pass

    @abc.abstractmethod
    def stack(
        self, 
        call_outputs: list[tensors.GraphTensor]
    ) -> tensors.GraphTensor:
        pass 

    def get_config(self) -> dict:
        return {}
    
    @classmethod
    def from_config(cls, config: dict) -> 'Featurizer':
        return cls(**config)
    
    def save(self, filepath: str | Path, *args, **kwargs) -> None:
        save_featurizer(
            self, filepath, *args, **kwargs
        )

    @staticmethod
    def load(filepath: str | Path, *args, **kwargs) -> 'Featurizer':
        return load_featurizer(
            filepath, *args, **kwargs
        )
    
    def __call__(
        self,
        inputs: str | tuple | list | np.ndarray | pd.DataFrame | pd.Series,
        *,
        multiprocessing: bool = False,
        processes: int | None = None,
        device: str = '/cpu:0',
        **kwargs
    ) -> tensors.GraphTensor:
        if isinstance(inputs, (str, tuple)):
            return self.call(inputs)
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            inputs = inputs.values
        if isinstance(inputs, np.ndarray):
            inputs = list(inputs)
        if not multiprocessing:
            outputs = [self.call(x) for x in inputs]
        else:
            with tf.device(device):
                with mp.Pool(processes) as pool:
                    outputs = pool.map(func=self.call, iterable=inputs)
        outputs = [x for x in outputs if x is not None]
        return self.stack(outputs)


@keras.saving.register_keras_serializable(package='molcraft')
class MolGraphFeaturizer(Featurizer):

    """Molecular graph featurizer.

    Converts SMILES or InChI strings to a molecular graph.

    The molecular graph may encode a single molecule or a batch of molecules.
    In either case, it is a single graph, with each molecule corresponding to
    a subgraph within the graph.

    Example:

    >>> import molcraft 
    >>> 
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer(
    ...     atom_features=[
    ...         molcraft.features.AtomType(),
    ...         molcraft.features.TotalNumHs(),
    ...         molcraft.features.Degree(),
    ...     ],
    ...     radius=1
    ... )
    >>> 
    >>> graph = featurizer(["N[C@@H](C)C(=O)O", "N[C@@H](CS)C(=O)O"])
    >>> graph
    GraphTensor(
        context={
            'size': <tf.Tensor: shape=[2], dtype=int32>
        },
        node={
            'feature': <tf.Tensor: shape=[13, 133], dtype=float32>
        },
        edge={
            'source': <tf.Tensor: shape=[22], dtype=int32>,
            'target': <tf.Tensor: shape=[22], dtype=int32>,
            'feature': <tf.Tensor: shape=[22, 5], dtype=float32>
        }
    )

    Args:
        atom_features:
            A list of `features.Feature` encoding the nodes of the molecular graph.
        bond_features:
            A list of `features.Feature` encoding the edges of the molecular graph.
        molecule_features:
            A `features.Feature` encoding the molecule (or `context`) of the graph.
            If `contextual_super_atom` is set to `True`, then this feature will be 
            embedded, via `NodeEmbedding`, as a super node in the molecular graph.
        super_atom:
            A boolean specifying whether super atoms exist and should be embedded 
            via `NodeEmbedding`.
        radius:
            An integer specifying how many bond lengths should be considered as an 
            edge. The default is None (or 1), which specifies that only bonds should 
            be considered an edge.
        self_loops:
            A boolean specifying whether self loops exist. If True, this means that
            each node (atom) has an edge (bond) to itself.
        include_hs:
            A boolean specifying whether hydrogens should be encoded as nodes.
    """

    def __init__(
        self,
        atom_features: list[features.Feature] | str | None = 'auto',
        bond_features: list[features.Feature] | str | None = 'auto',
        molecule_features: features.Feature | str | None = None,
        super_atom: bool = False,
        radius: int | float | None = None,
        self_loops: bool = False,
        include_hs: bool = False,
        **kwargs,
    ) -> None:
        if molecule_features is None:
            molecule_features = kwargs.pop('mol_features', None)
            
        self.radius = int(max(radius or 1, 1))
        self.include_hs = include_hs
        self.self_loops = self_loops
        self.super_atom = super_atom

        default_atom_features = (
            atom_features == 'auto' or atom_features == 'default'
        )
        if default_atom_features:
            atom_features = [features.AtomType()]
            if not self.include_hs:
                atom_features.append(features.TotalNumHs())
            atom_features.append(features.Degree())
        if not isinstance(self, MolGraphFeaturizer3D):
            default_bond_features = (
                bond_features == 'auto' or bond_features == 'default'
            )
            if default_bond_features or self.radius > 1:
                vocab = ['zero', 'single', 'double', 'triple', 'aromatic']
                bond_features = [
                    features.BondType(vocab)
                ]
        default_molecule_features = (
            molecule_features == 'auto' or molecule_features == 'default'
        )
        if default_molecule_features:
            molecule_features = [
                descriptors.MolWeight(),
                descriptors.TPSA(),
                descriptors.CrippenLogP(),
                descriptors.CrippenMolarRefractivity(),
                descriptors.NumHeavyAtoms(),
                descriptors.NumHeteroAtoms(),
                descriptors.NumHydrogenDonors(),
                descriptors.NumHydrogenAcceptors(),
                descriptors.NumRotatableBonds(),
                descriptors.NumRings(),
            ]
        self._atom_features = atom_features
        self._bond_features = bond_features
        self._molecule_features = molecule_features
        self.feature_dtype = 'float32'
        self.index_dtype = 'int32'

    def call(self, inputs: str | tuple) -> tensors.GraphTensor:
        if isinstance(inputs, (tuple, list, np.ndarray)):
            x, *context = inputs
            if len(context) and isinstance(context[0], dict):
                context = copy.deepcopy(context[0])
        else:
            x, context = inputs, None

        mol = chem.Mol.from_encoding(x, explicit_hs=self.include_hs)

        if mol is None:
            warn(
                f'Could not obtain `chem.Mol` from {x}. '
                'Returning `None` (proceeding without it).'
            )
            return None
        
        atom_feature = self.atom_features(mol)
        bond_feature = self.bond_features(mol)
        molecule_feature = self.molecule_feature(mol)
        molecule_size = self.num_atoms(mol)
        
        if isinstance(context, dict):
            if 'x' in context:
                context['feature'] = context.pop('x')
            if 'y' in context:
                context['label'] = context.pop('y')
            if 'sample_weight' in context:
                context['weight'] = context.pop('sample_weight')
            context = {
                **{'size': molecule_size}, 
                **context
            }
        elif isinstance(context, list):
            context = {
                **{'size': molecule_size}, 
                **{key: value for (key, value) in zip(['label', 'weight'], context)}
            }
        else:
            context = {'size': molecule_size}

        if molecule_feature is not None:
            if 'feature' in context:
                warn(
                    'Found both inputted and computed context feature. '
                    'Overwriting inputted context feature with computed '
                    'context feature (based on `molecule_features`).'
                )
            context['feature'] = molecule_feature

        node = {}
        node['feature'] = atom_feature
        
        edge = {}
        if self.radius == 1:
            edge['source'], edge['target'] = mol.adjacency(
                fill='full', sparse=True, self_loops=self.self_loops, dtype=self.index_dtype
            )
            if self.self_loops:
                bond_feature = np.pad(bond_feature, [(0, 1), (0, 0)])
            if bond_feature is not None:
                bond_indices = []
                for atom_i, atom_j in zip(edge['source'], edge['target']):
                    if atom_i == atom_j:
                        bond_indices.append(-1)
                    else:
                        bond_indices.append(
                            mol.get_bond_between_atoms(atom_i, atom_j).index
                        )
                edge['feature'] = bond_feature[bond_indices]
                if self.self_loops:
                    edge['self_loop'] = (edge['source'] == edge['target'])
        else:
            paths = chem.get_shortest_paths(
                mol, radius=self.radius, self_loops=self.self_loops
            )
            edge['source'] = np.asarray(
                [path[0] for path in paths], dtype=self.index_dtype
            )
            edge['target'] = np.asarray(
                [path[-1] for path in paths], dtype=self.index_dtype
            )
            edge['length'] = np.asarray(
                [len(path) - 1 for path in paths], dtype=self.index_dtype
            )
            if bond_feature is not None:
                zero_bond_feature = np.array(
                    [[1., 0., 0., 0., 0.]], dtype=bond_feature.dtype
                )
                bond_feature = np.concatenate(
                    [bond_feature, zero_bond_feature], axis=0
                )    
                edge['feature'] = self._expand_bond_features(
                    mol, paths, bond_feature,
                )
            edge['length'] = np.eye(self.radius + 1, dtype=self.feature_dtype)[edge['length']]

        if self.super_atom:
            node, edge = self._add_super_atom(node, edge)
            context['size'] += 1

        return tensors.GraphTensor(context, node, edge)

    def stack(self, outputs):
        if tensors.is_scalar(outputs[0]):
            return tf.stack(outputs, axis=0)
        return tf.concat(outputs, axis=0)
    
    def atom_features(self, mol: chem.Mol) -> np.ndarray:
        atom_feature: np.ndarray = np.concatenate(
            [f(mol) for f in self._atom_features], axis=-1
        )
        return atom_feature.astype(self.feature_dtype)

    def bond_features(self, mol: chem.Mol) -> np.ndarray:
        if self._bond_features is None:
            return None
        bond_feature: np.ndarray = np.concatenate(
            [f(mol) for f in self._bond_features], axis=-1
        )
        return bond_feature.astype(self.feature_dtype)
    
    def molecule_feature(self, mol: chem.Mol) -> np.ndarray:
        if self._molecule_features is None:
            return None
        molecule_feature: np.ndarray = np.concatenate(
            [f(mol) for f in self._molecule_features], axis=-1
        )
        return molecule_feature.astype(self.feature_dtype)

    def num_atoms(self, mol: chem.Mol) -> np.ndarray:
        return np.asarray(mol.num_atoms, dtype=self.index_dtype)
    
    def num_bonds(self, mol: chem.Mol) -> np.ndarray:
        return np.asarray(mol.num_bonds, dtype=self.index_dtype) 

    def _expand_bond_features(
        self, 
        mol: chem.Mol,
        paths: list[list[int]],
        bond_feature: np.ndarray,
    ) -> np.ndarray:  

        def bond_feature_lookup(path):
            path_bond_indices = [
                mol.get_bond_between_atoms(path[i], path[i + 1]).index
                for i in range(len(path) - 1)
            ]
            padding = [-1] * (self.radius - len(path) + 1)
            path_bond_indices += padding 
            return bond_feature[path_bond_indices].reshape(-1)

        edge_feature = np.asarray(
            [
                bond_feature_lookup(path) for path in paths
            ], 
            dtype=self.feature_dtype
        ).reshape((-1, bond_feature.shape[-1] * self.radius))

        return edge_feature
        
    def _add_super_atom(
        self, 
        node: dict[str, np.ndarray],
        edge: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray]]:
        num_super_nodes = 1 
        num_nodes = node['feature'].shape[0]
        node = _add_super_nodes(node, num_super_nodes)
        edge = _add_super_edges(
            edge, num_nodes, num_super_nodes, self.feature_dtype, self.index_dtype
        )
        return node, edge

    def get_config(self):
        config = super().get_config()
        config.update({
            'atom_features': keras.saving.serialize_keras_object(
                self._atom_features
            ),
            'bond_features': keras.saving.serialize_keras_object(
                self._bond_features
            ),
            'molecule_features': keras.saving.serialize_keras_object(
                self._molecule_features
            ),
            'super_atom': self.super_atom,
            'radius': self.radius,
            'self_loops': self.self_loops,
            'include_hs': self.include_hs,
        })
        return config

    @classmethod
    def from_config(cls, config: dict):
        config['atom_features'] = keras.saving.deserialize_keras_object(
            config['atom_features']
        )
        config['bond_features'] = keras.saving.deserialize_keras_object(
            config['bond_features']
        )
        config['molecule_features'] = keras.saving.deserialize_keras_object(
            config['molecule_features']
        )
        return cls(**config)
    

@keras.saving.register_keras_serializable(package='molcraft')
class MolGraphFeaturizer3D(MolGraphFeaturizer):

    """Molecular 3d-graph featurizer.

    Converts SMILES or InChI strings to a molecular graph in 3d space.
    Namely, in addition to the information encoded in a standard molecular
    graph, cartesian coordinates are also included. 

    The molecular graph may encode a single molecule or a batch of molecules.
    In either case, it is a single graph, with each molecule corresponding to
    a subgraph within the graph.

    Example:

    >>> import molcraft 
    >>> 
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer3D(
    ...     atom_features=[
    ...         molcraft.features.AtomType(),
    ...         molcraft.features.TotalNumHs(),
    ...         molcraft.features.Degree(),
    ...     ],
    ...     radius=5.0
    ... )
    >>> 
    >>> graph = featurizer(["N[C@@H](C)C(=O)O", "N[C@@H](CS)C(=O)O"])
    >>> graph
    GraphTensor(
        context={
            'size': <tf.Tensor: shape=[20], dtype=int32>
        },
        node={
            'feature': <tf.Tensor: shape=[130, 133], dtype=float32>,
            'coordinate': <tf.Tensor: shape=[130, 3], dtype=float32>
        },
        edge={
            'source': <tf.Tensor: shape=[714], dtype=int32>,
            'target': <tf.Tensor: shape=[714], dtype=int32>,
            'feature': <tf.Tensor: shape=[714, 23], dtype=float32>
        }
    )
        
    Args:
        atom_features:
            A list of `features.Feature` encoding the nodes of the molecular graph.
        bond_features:
            A list of `features.Feature` encoding the edges of the molecular graph.
        molecule_features:
            A `features.Feature` encoding the molecule (or `context`) of the graph.
            If `contextual_super_atom` is set to `True`, then this feature will be 
            embedded, via `NodeEmbedding`, as a super node in the molecular graph.
        conformer_generator:
            A `conformers.ConformerGenerator` which produces conformers. If `auto`
            a `conformers.ConformerEmbedder` will be used. If None, it is assumed
            that the molecule already has conformer(s).
        super_atom:
            A boolean specifying whether super atoms exist and should be embedded 
            via `NodeEmbedding`.
        radius:
            A float specifying, for each atom, the maximum distance (in angstroms)
            that another atom should be within to be considered an edge. Default
            is set to 6.0 as this should cover most interactions. This parameter
            can be though of as the receptive field. If None, the radius will be
            infinite so all the receptive field will be the entire space (graph).
        self_loops:
            A boolean specifying whether self loops exist. If True, this means that
            each node (atom) has an edge (bond) to itself.
        include_hs:
            A boolean specifying whether hydrogens should be encoded as nodes.
    """

    def __init__(
        self,
        atom_features: list[features.Feature] | str | None = 'auto',
        bond_features: list[features.Feature] | str | None = 'auto',
        molecule_features: features.Feature | str = None,
        conformer_generator: conformers.ConformerProcessor | str | None = 'auto',
        super_atom: bool = False,
        radius: int | float | None = 6.0,
        self_loops: bool = False,
        include_hs: bool = False,
        **kwargs,
    ) -> None:
        if bond_features == 'auto':
            bond_features = [
                features.Distance()
            ]
        super().__init__(
            atom_features=atom_features,
            bond_features=bond_features,
            molecule_features=molecule_features,
            super_atom=super_atom,
            radius=radius,
            self_loops=self_loops,
            include_hs=include_hs,
            **kwargs,
        )
        if conformer_generator == 'auto':
            conformer_generator = conformers.ConformerGenerator(
                steps=[
                    conformers.ConformerEmbedder(
                        method='ETKDGv3', 
                        num_conformers=5
                    ),
                ]
            )
        self.conformer_generator = conformer_generator
        self.embed_conformer = self.conformer_generator is not None
        self.radius = float(radius) if radius else None
        
    def call(self, inputs: str | tuple) -> tensors.GraphTensor:

        if isinstance(inputs, (tuple, list, np.ndarray)):
            x, *context = inputs
            if len(context) and isinstance(context[0], dict):
                context = copy.deepcopy(context[0])
        else:
            x, context = inputs, None

        explicit_hs = (self.include_hs or self.embed_conformer)
        mol = chem.Mol.from_encoding(x, explicit_hs=explicit_hs)
        
        if mol is None:
            warn(
                f'Could not obtain `chem.Mol` from {x}. '
                'Proceeding without it.'
            )
            return None

        if self.embed_conformer:
            mol = self.conformer_generator(mol)
            if not self.include_hs:
                mol = chem.remove_hs(mol)

        if mol.num_conformers == 0:
            raise ValueError(
                'Cannot featurize a molecule without conformer(s). '
                'Make sure to pass a `ConformerGenerator` to the constructor '
                'of the `Featurizer` or input a 3D representation of the molecule. '
            )

        molecule_feature = self.molecule_feature(mol)
        molecule_size = self.num_atoms(mol) + int(self.super_atom)

        if isinstance(context, dict):
            if 'x' in context:
                context['feature'] = context.pop('x')
            if 'y' in context:
                context['label'] = context.pop('y')
            if 'sample_weight' in context:
                context['weight'] = context.pop('sample_weight')
            context = {
                **{'size': molecule_size}, 
                **context
            }
        elif isinstance(context, list):
            context = {
                **{'size': molecule_size}, 
                **{key: value for (key, value) in zip(['label', 'weight'], context)}
            }
        else:
            context = {'size': molecule_size}

        if molecule_feature is not None:
            if 'feature' in context:
                warn(
                    'Found both inputted and computed context feature. '
                    'Overwriting inputted context feature with computed '
                    'context feature (based on `molecule_features`).'
                )
            context['feature'] = molecule_feature
            
        node = {}
        node['feature'] = self.atom_features(mol)

        if self._bond_features:
            edge_feature = self.bond_features(mol)

        edge = {}
        mols = chem.unpack_conformers(mol)
        tensor_list = []
        for i, mol in enumerate(mols):
            node_conformer = copy.deepcopy(node)
            edge_conformer = copy.deepcopy(edge)
            conformer = mol.get_conformer()
            adjacency_matrix = conformer.adjacency(
                fill='full', 
                radius=self.radius, 
                sparse=False, 
                self_loops=self.self_loops, 
                dtype=bool
            )
            edge_conformer['source'], edge_conformer['target'] = np.where(adjacency_matrix)
            edge_conformer['source'] = edge_conformer['source'].astype(self.index_dtype)
            edge_conformer['target'] = edge_conformer['target'].astype(self.index_dtype)
            node_conformer['coordinate'] = conformer.coordinates.astype(self.feature_dtype)

            if self._bond_features:
                edge_feature_keep = adjacency_matrix.reshape(-1)
                edge_conformer['feature'] = edge_feature[edge_feature_keep]

            if self.super_atom:
                node_conformer, edge_conformer = self._add_super_atom(
                    node_conformer, edge_conformer
                )
                node_conformer['coordinate'] = np.concatenate(
                    [node_conformer['coordinate'], conformer.centroid[None]], axis=0
                ).astype(self.feature_dtype)
            tensor_list.append(
                tensors.GraphTensor(context, node_conformer, edge_conformer)
            )
            
        return tensor_list
    
    def stack(self, outputs):
        # Flatten list of lists (due to multiple conformers per molecule)
        outputs = [x for xs in outputs for x in xs]
        return super().stack(outputs)
    
    def get_config(self):
        config = super().get_config()
        config['conformer_generator'] = keras.saving.serialize_keras_object(
            self.conformer_generator
        )
        return config

    @classmethod
    def from_config(cls, config: dict):
        config['conformer_generator'] = keras.saving.deserialize_keras_object(
            config['conformer_generator']
        )
        return super().from_config(config)
    

def save_featurizer(
    featurizer: Featurizer, 
    filepath: str | Path, 
    overwrite: bool = True, 
    **kwargs
) -> None:
    filepath = Path(filepath)
    if filepath.suffix != '.json':
        raise ValueError(
            'Invalid `filepath` extension for saving a `Featurizer`. '
            'A `Featurizer` should be saved as a JSON file.'
        )
    if not filepath.parent.exists():
        filepath.parent.mkdir(parents=True, exist_ok=True)
    if filepath.exists() and not overwrite:
        return 
    serialized_featurizer = keras.saving.serialize_keras_object(featurizer)
    with open(filepath, 'w') as f:
        json.dump(serialized_featurizer, f, indent=4)

def load_featurizer(
    filepath: str | Path,
    **kwargs
) -> Featurizer:
    filepath = Path(filepath)
    if filepath.suffix != '.json':
        raise ValueError(
            'Invalid `filepath` extension for loading a `Featurizer`. '
            'A `Featurizer` should be saved as a JSON file.'
        )
    if not filepath.exists():
        return 
    with open(filepath, 'r') as f:
        config = json.load(f)
    return keras.saving.deserialize_keras_object(config)

def _add_super_nodes(
    node: dict[str, np.ndarray], 
    num_super_nodes: int = 1,
) -> dict[str, np.ndarray]:
    node = copy.deepcopy(node)
    node['super'] = np.array(
        [False] * len(node['feature']) + [True] * num_super_nodes,
        dtype=bool
    )
    super_node_feature = np.zeros(
        [num_super_nodes, node['feature'].shape[-1]], 
        dtype=node['feature'].dtype
    )
    node['feature'] = np.concatenate([node['feature'], super_node_feature])
    return node 

def _add_super_edges(
    edge: dict[str, np.ndarray], 
    num_nodes: int,
    num_super_nodes: int,
    feature_dtype: str,
    index_dtype: str,
) -> dict[str, np.ndarray]:
    edge = copy.deepcopy(edge)
    super_node_indices = (
        np.repeat(np.arange(num_super_nodes), [num_nodes]) + num_nodes
    )
    node_indices = (
        np.tile(np.arange(num_nodes), [num_super_nodes])
    )
    edge['source'] = np.concatenate(
        [edge['source'], node_indices, super_node_indices]
    ).astype(index_dtype)
   
    edge['target'] = np.concatenate(
        [edge['target'], super_node_indices, node_indices]
    ).astype(index_dtype)

    if 'feature' in edge:
        num_edges = int(edge['feature'].shape[0])
        num_super_edges = int(num_super_nodes * num_nodes * 2)
        edge['super'] = np.asarray(
            ([False] * num_edges + [True] * num_super_edges),
            dtype=bool
        )
        edge['feature'] = np.concatenate(
            [
                edge['feature'], 
                np.zeros(
                    shape=(num_super_edges, edge['feature'].shape[-1]),
                    dtype=edge['feature'].dtype
                )
            ]
        )

    if 'self_loop' in edge:
        edge['self_loop'] = np.pad(
            edge['self_loop'], [(0, num_nodes * num_super_nodes * 2)],
            constant_values=False,
        )
    if 'length' in edge:
        edge['length'] = np.pad(edge['length'], [(0, 0), (1, 0)])
        zero_array = np.zeros([num_nodes * num_super_nodes * 2], dtype='int32')
        edge_length_dim = edge['length'].shape[1]
        virtual_edge_length = np.eye(edge_length_dim)[zero_array]
        edge['length'] = np.concatenate([edge['length'], virtual_edge_length])
        edge['length'] = edge['length'].astype(feature_dtype)

    return edge

    
def warn(message: str) -> None:
    warnings.warn(
        message=message,
        category=UserWarning,
        stacklevel=1
    )

MolFeaturizer = MolGraphFeaturizer
MolFeaturizer3D = MolGraphFeaturizer3D