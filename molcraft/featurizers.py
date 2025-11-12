import warnings
import keras 
import json
import abc
import typing 
import numpy as np
import pandas as pd
import tensorflow as tf
import multiprocessing as mp

from pathlib import Path

from molcraft import tensors 
from molcraft import features
from molcraft import records
from molcraft import chem
from molcraft import descriptors


@keras.saving.register_keras_serializable(package='molcraft')
class GraphFeaturizer(abc.ABC):

    """Base graph featurizer.
    """

    @abc.abstractmethod
    def call(self, x: str | chem.Mol | tuple) -> tensors.GraphTensor:
        pass

    def get_config(self) -> dict:
        return {}
    
    @classmethod
    def from_config(cls, config: dict) -> 'GraphFeaturizer':
        return cls(**config)
    
    def save(self, filepath: str | Path, *args, **kwargs) -> None:
        save_featurizer(self, filepath, *args, **kwargs)

    @staticmethod
    def load(filepath: str | Path, *args, **kwargs) -> 'GraphFeaturizer':
        return load_featurizer(filepath, *args, **kwargs)
    
    def write_records(self, inputs: str | chem.Mol | tuple, path: str | Path, **kwargs) -> None:
         records.write(
            inputs, featurizer=self, path=path, **kwargs
         )

    @staticmethod
    def read_records(path: str | Path, **kwargs) -> tf.data.Dataset:
        return records.read(
            path=path, **kwargs
        )
        
    def __call__(
        self,
        inputs: str | chem.Mol | tuple | typing.Iterable,
        *,
        multiprocessing: bool = False,
        processes: int | None = None,
        device: str = '/cpu:0',
    ) -> tensors.GraphTensor:
        if isinstance(inputs, (str, tuple)):
            return self.call(inputs)
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            inputs = inputs.values.tolist()
        elif isinstance(inputs, np.ndarray):
            inputs = inputs.tolist()
        if not multiprocessing:
            outputs = [self.call(x) for x in inputs]
        else:
            with tf.device(device):
                with mp.Pool(processes) as pool:
                    outputs = pool.map(func=self.call, iterable=inputs)
        outputs = [x for x in outputs if x is not None]
        if tensors.is_scalar(outputs[0]):
            return tf.stack(outputs, axis=0)
        return tf.concat(outputs, axis=0)


@keras.saving.register_keras_serializable(package='molcraft')
class MolGraphFeaturizer(GraphFeaturizer):

    """Molecular graph featurizer.

    Converts SMILES or InChI strings to a molecular graph.

    The molecular graph may encode a single molecule or a batch of molecules.

    Example:

    >>> import molcraft 
    >>> 
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer(
    ...     atom_features=[
    ...         molcraft.features.AtomType(),
    ...         molcraft.features.NumHydrogens(),
    ...         molcraft.features.Degree(),
    ...     ],
    ...     bond_features=[
    ...         molcraft.features.BondType(),
    ...     ],
    ...     super_node=False,
    ...     self_loops=False,
    ... )
    >>> 
    >>> graph = featurizer(["N[C@@H](C)C(=O)O", "N[C@@H](CS)C(=O)O"])
    >>> graph
    GraphTensor(
        context={
            'size': <tf.Tensor: shape=[2], dtype=int32>
        },
        node={
            'feature': <tf.Tensor: shape=[13, 129], dtype=float32>
        },
        edge={
            'source': <tf.Tensor: shape=[22], dtype=int32>,
            'target': <tf.Tensor: shape=[22], dtype=int32>,
            'feature': <tf.Tensor: shape=[22, 5], dtype=float32>
        }
    )

    Args:
        atom_features:
            A list of `features.Feature` encoded as the node features.
        bond_features:
            A list of `features.Feature` encoded as the edge features.
        molecule_features:
            A list of `descriptors.Descriptor` encoded as the context feature.
        super_node:
            A boolean specifying whether to include a super node.
        self_loops:
            A boolean specifying whether self loops exist.
        include_hydrogens:
            A boolean specifying whether hydrogens should be encoded as nodes.
        wildcards:
            A boolean specifying whether wildcards exist. If True, wildcard labels will
            be encoded in the graph and separately embedded in `layers.NodeEmbedding`.
    """

    def __init__(
        self,
        atom_features: list[features.Feature] | str = 'auto',
        bond_features: list[features.Feature] | str | None = 'auto',
        molecule_features: list[descriptors.Descriptor] | str | None = None,
        super_node: bool = False,
        self_loops: bool = False,
        include_hydrogens: bool = False,
        wildcards: bool = False,
    ) -> None:
        use_default_atom_features = (
            atom_features == 'auto' or atom_features == 'default'
        )
        if use_default_atom_features:
            atom_features = [features.AtomType(), features.Degree()]
            if not include_hydrogens:
                atom_features += [features.NumHydrogens()]

        use_default_bond_features = (
            bond_features == 'auto' or bond_features == 'default'
        )
        if use_default_bond_features:
            bond_features = [features.BondType()]

        use_default_molecule_features = (
            molecule_features == 'auto' or molecule_features == 'default'
        )
        if use_default_molecule_features:
            molecule_features = [
                descriptors.MolWeight(),
                descriptors.TotalPolarSurfaceArea(),
                descriptors.LogP(),
                descriptors.MolarRefractivity(),
                descriptors.NumHeavyAtoms(),
                descriptors.NumHeteroatoms(),
                descriptors.NumHydrogenDonors(),
                descriptors.NumHydrogenAcceptors(),
                descriptors.NumRotatableBonds(),
                descriptors.NumRings(),
            ]

        self._atom_features = atom_features
        self._bond_features = bond_features
        self._molecule_features = molecule_features
        self._include_hydrogens = include_hydrogens
        self._wildcards = wildcards
        self._self_loops = self_loops
        self._super_node = super_node

    def call(self, inputs: str | chem.Mol | tuple) -> tensors.GraphTensor:
        
        if isinstance(inputs, (str, chem.Mol, chem.RDKitMol)):
            inputs = (inputs,)

        inputs, *context_inputs = inputs

        if isinstance(inputs, str):
            mol = chem.Mol.from_encoding(
                inputs, explicit_hs=self._include_hydrogens
            )
        elif isinstance(inputs, chem.RDKitMol):
            mol = chem.Mol.cast(inputs)
        else:
            mol = inputs

        data = {'context': {}, 'node': {}, 'edge': {}}
        
        data['context']['size'] = np.asarray(mol.num_atoms)

        if len(context_inputs) == 1:
            data['context']['label'] = np.asarray(context_inputs[0])
        elif len(context_inputs) == 2:
            data['context']['label'] = np.asarray(context_inputs[0])
            data['context']['sample_weight'] = np.asarray(context_inputs[1])

        if self._molecule_features is not None:
            data['context']['feature'] = np.concatenate(
                [f(mol) for f in self._molecule_features], axis=-1
            )

        data['node']['feature'] = np.concatenate(
            [f(mol) for f in self._atom_features], axis=-1
        )

        if self._wildcards:
            wildcard_labels = np.asarray([
                (atom.label or 0) + 1 if atom.symbol == "*" else 0
                for atom in mol.atoms
            ])
            data['node']['wildcard'] = wildcard_labels
            data['node']['feature'] = np.where(
                wildcard_labels[:, None],
                np.zeros_like(data['node']['feature']),
                data['node']['feature']
            )

        data['edge']['source'], data['edge']['target'] = mol.adjacency(
            fill='full', sparse=True, self_loops=self._self_loops
        )
        
        if self._bond_features is not None:
            bond_features = np.concatenate(
                [f(mol) for f in self._bond_features], axis=-1
            )
            if self._self_loops:
                bond_features = np.pad(bond_features, [(0, 1), (0, 0)])

            bond_indices = [
                mol.get_bond_between_atoms(i, j).index if (i != j) else -1
                for (i, j) in zip(data['edge']['source'], data['edge']['target'])
            ]

            data['edge']['feature'] = bond_features[bond_indices]

        if self._super_node:
            data = _add_super_node(data)

        return tensors.GraphTensor(**_convert_dtypes(data))
    
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
            'super_node': self._super_node,
            'self_loops': self._self_loops,
            'include_hydrogens': self._include_hydrogens,
            'wildcards': self._wildcards,
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

    """3D Molecular graph featurizer.

    Converts SMILES or InChI strings to a 3d molecular graph.

    The molecular graph may encode a single molecule or a batch of molecules.

    Example:

    >>> import molcraft 
    >>> 
    >>> featurizer = molcraft.featurizers.MolGraphFeaturizer3D(
    ...     atom_features=[
    ...         molcraft.features.AtomType(),
    ...         molcraft.features.NumHydrogens(),
    ...         molcraft.features.Degree(),
    ...     ],
    ...     radius=5.0,
    ...     random_seed=42,
    ... )
    >>> 
    >>> graph = featurizer(["N[C@@H](C)C(=O)O", "N[C@@H](CS)C(=O)O"])
    >>> graph
    GraphTensor(
        context={
            'size': <tf.Tensor: shape=[2], dtype=int32>
        },
        node={
            'feature': <tf.Tensor: shape=[13, 129], dtype=float32>,
            'coordinate': <tf.Tensor: shape=[13, 3], dtype=float32>
        },
        edge={
            'source': <tf.Tensor: shape=[72], dtype=int32>,
            'target': <tf.Tensor: shape=[72], dtype=int32>,
            'feature': <tf.Tensor: shape=[72, 12], dtype=float32>
        }
    )
        
    Args:
        atom_features:
            A list of `features.Feature` encoded as the node features.
        pair_features:
            A list of `features.PairFeature` encoded as the edge features.
        molecule_features:
            A list of `descriptors.Descriptor` encoded as the context feature.
        super_node:
            A boolean specifying whether to include a super node.
        self_loops:
            A boolean specifying whether self loops exist.
        include_hydrogens:
            A boolean specifying whether hydrogens should be encoded as nodes.
        wildcards:
            A boolean specifying whether wildcards exist. If True, wildcard labels will
            be encoded in the graph and separately embedded in `layers.NodeEmbedding`.
        radius:
            A floating point value specifying maximum edge length. 
        random_seed:
            An integer specifying the random seed for the conformer generation.
    """

    def __init__(
        self,
        atom_features: list[features.Feature] | str = 'auto',
        pair_features: list[features.PairFeature] | str = 'auto',
        molecule_features: features.Feature | str | None = None,
        super_node: bool = False,
        self_loops: bool = False,
        include_hydrogens: bool = False,
        wildcards: bool = False,
        radius: int | float | None = 6.0,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        kwargs.pop('bond_features', None)
        super().__init__(
            atom_features=atom_features,
            bond_features=None,
            molecule_features=molecule_features,
            super_node=super_node,
            self_loops=self_loops,
            include_hydrogens=include_hydrogens,
            wildcards=wildcards,
        )

        use_default_pair_features = (
            pair_features == 'auto' or pair_features == 'default'
        )
        if use_default_pair_features:
            pair_features = [features.PairDistance()]

        self._pair_features = pair_features 
        self._radius = float(radius) if radius else None
        self._random_seed = random_seed

    def call(self, inputs: str | tuple) -> tensors.GraphTensor:

        if isinstance(inputs, (str, chem.Mol, chem.RDKitMol)):
            inputs = (inputs,)

        inputs, *context_inputs = inputs

        if isinstance(inputs, str):
            mol = chem.Mol.from_encoding(inputs, explicit_hs=True)
        elif isinstance(inputs, chem.RDKitMol):
            mol = chem.Mol.cast(inputs)
        else:
            mol = inputs

        if mol.num_conformers == 0:
            mol = chem.embed_conformers(
                mol, num_conformers=1, random_seed=self._random_seed
            )

        if not self._include_hydrogens:
            mol = chem.remove_hs(mol)

        data = {'context': {}, 'node': {}, 'edge': {}}

        data['context']['size'] = np.asarray(mol.num_atoms)

        if len(context_inputs) == 1:
            data['context']['label'] = np.asarray(context_inputs[0])
        elif len(context_inputs) == 2:
            data['context']['label'] = np.asarray(context_inputs[0])
            data['context']['sample_weight'] = np.asarray(context_inputs[1])

        if self._molecule_features is not None:
            data['context']['feature'] = np.concatenate(
                [f(mol) for f in self._molecule_features], axis=-1
            )

        conformer = mol.get_conformer()

        data['node']['feature'] = np.concatenate(
            [f(mol) for f in self._atom_features], axis=-1
        )
        data['node']['coordinate'] = conformer.coordinates

        if self._wildcards:
            wildcard_labels = np.asarray([
                (atom.label or 0) + 1 if atom.symbol == "*" else 0
                for atom in mol.atoms
            ])
            data['node']['wildcard'] = wildcard_labels
            data['node']['feature'] = np.where(
                wildcard_labels[:, None],
                np.zeros_like(data['node']['feature']),
                data['node']['feature']
            )

        adjacency_matrix = conformer.adjacency(
            fill='full', radius=self._radius, sparse=False, self_loops=self._self_loops,
        )

        data['edge']['source'], data['edge']['target'] = np.where(adjacency_matrix)
        
        if self._pair_features is not None:
            pair_features = np.concatenate(
                [f(mol) for f in self._pair_features], axis=-1
            )
            pair_keep = adjacency_matrix.reshape(-1).astype(bool)
            data['edge']['feature'] = pair_features[pair_keep]

        if self._super_node:
            data = _add_super_node(data)
            data['node']['coordinate'] = np.concatenate(
                [data['node']['coordinate'], conformer.centroid[None]], axis=0
            )

        return tensors.GraphTensor(**_convert_dtypes(data))
    
    @property 
    def random_seed(self) -> int | None:
        return self._random_seed
    
    @random_seed.setter
    def random_seed(self, value: int) -> None:
        self._random_seed = value 

    def get_config(self):
        config = super().get_config()
        config['radius'] = self._radius
        config['pair_features'] = keras.saving.serialize_keras_object(
            self._pair_features
        )
        config['random_seed'] = self._random_seed
        return config

    @classmethod
    def from_config(cls, config: dict):
        config['pair_features'] = keras.saving.deserialize_keras_object(
            config['pair_features']
        )
        return super().from_config(config)


def save_featurizer(
    featurizer: GraphFeaturizer, 
    filepath: str | Path, 
    overwrite: bool = True, 
    **kwargs
) -> None:
    filepath = Path(filepath)
    if filepath.suffix != '.json':
        raise ValueError(
            'Invalid `filepath` extension for saving a `GraphFeaturizer`. '
            'A `GraphFeaturizer` should be saved as a JSON file.'
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
) -> GraphFeaturizer:
    filepath = Path(filepath)
    if filepath.suffix != '.json':
        raise ValueError(
            'Invalid `filepath` extension for loading a `GraphFeaturizer`. '
            'A `GraphFeaturizer` should be saved as a JSON file.'
        )
    if not filepath.exists():
        return 
    with open(filepath, 'r') as f:
        config = json.load(f)
    return keras.saving.deserialize_keras_object(config)

def _add_super_node(
    data: dict[str, dict[str, np.ndarray]]
) -> dict[str, dict[str, np.ndarray]]:

    data['context']['size'] += 1

    num_nodes = data['node']['feature'].shape[0]
    num_edges = data['edge']['source'].shape[0]
    super_node_index = num_nodes

    add_self_loops = np.any(
        data['edge']['source'] == data['edge']['target']
    )
    if add_self_loops:
        data['edge']['source'] = np.append(
            data['edge']['source'], super_node_index
        )
        data['edge']['target'] = np.append(
            data['edge']['target'], super_node_index
        )

    data['node']['feature'] = np.pad(data['node']['feature'], [(0, 1), (0, 0)])
    data['node']['super'] = np.asarray([False] * num_nodes + [True])
    if 'wildcard' in data['node']:
        data['node']['wildcard'] = np.pad(data['node']['wildcard'], [(0, 1)])

    node_indices = list(range(num_nodes))
    super_node_indices = [super_node_index] * num_nodes

    data['edge']['source'] = np.append(
        data['edge']['source'], node_indices + super_node_indices
    )
    data['edge']['target'] = np.append(
        data['edge']['target'], super_node_indices + node_indices
    )
    
    total_num_edges = data['edge']['source'].shape[0]
    num_super_edges = (total_num_edges - num_edges) 
    data['edge']['super'] = np.asarray(
        [False] * num_edges + [True] * num_super_edges
    )

    if 'feature' in data['edge']:
        data['edge']['feature'] = np.pad(
            data['edge']['feature'], [(0, num_super_edges), (0, 0)]
        )
        
    return data

def _convert_dtypes(data: dict[str, dict[str, np.ndarray]]) -> np.ndarray:
    for outer_key, inner_dict in data.items():
        for inner_key, inner_value in inner_dict.items():
            if inner_key in ['source', 'target', 'size']:
                data[outer_key][inner_key] = inner_value.astype(np.int32)
            elif np.issubdtype(inner_value.dtype, np.integer):
                data[outer_key][inner_key] = inner_value.astype(np.int32)
            elif np.issubdtype(inner_value.dtype, np.floating):
                data[outer_key][inner_key] = inner_value.astype(np.float32)
    return data