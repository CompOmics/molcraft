import abc
import math
import keras
import warnings
import numpy as np

from molcraft import chem


@keras.saving.register_keras_serializable(package='molcraft')
class Feature(abc.ABC):

    def __init__(
        self, 
        vocab: set[int | str] = None, 
        allow_oov: bool = True,
        encode_oov: bool = False, 
        dtype: str = 'float32'
    ) -> None:
        self.encode_oov = encode_oov
        self.allow_oov = allow_oov
        self.oov_token = '<oov>'
        self.dtype = dtype 
        if not vocab:
            vocab = default_vocabulary.get(self.name, None)
        if vocab:
            if isinstance(vocab, set):
                vocab: list = list(vocab)
                vocab.sort(key=lambda x: x if x is not None else "")
            elif not isinstance(vocab, list):
                vocab: list = list(vocab)
            if self.encode_oov and self.oov_token not in vocab:
                vocab.append(self.oov_token)
            onehot_encodings = np.eye(len(vocab), dtype=self.dtype)
            self.feature_to_onehot = dict(zip(vocab, onehot_encodings))
        self.vocab = vocab 

    @abc.abstractmethod
    def call(self, mol: chem.Mol) -> list[float | int | bool | str]: 
        pass
    
    def __call__(self, mol: chem.Mol) -> np.ndarray:
        if not isinstance(mol, chem.Mol):
            raise ValueError(
                f'Input to {self.name} needs to be a `chem.Mol`, which '
                'implements two properties that should be iterated over '
                'to compute features: `atoms` and `bonds`.'
            )
        features = self.call(mol)
        if len(features) != mol.num_atoms and len(features) != mol.num_bonds:
            raise ValueError(
                f'The number of features computed by {self.name} does not '
                'match the number of atoms or bonds of the `chem.Mol` object. '
                'Make sure to iterate over `atoms` or `bonds` of `chem.Mol` '
                'when computing features.'
            )
        if len(features) == 0:
            # Edge case: no atoms or bonds in the molecule.
            return np.zeros((0, self.output_dim), dtype=self.dtype)
        
        func = (
            self._featurize_categorical if self.vocab else 
            self._featurize_floating
        )
        return np.stack([func(x) for x in features])
        

    def get_config(self) -> dict:
        config = {
            'vocab': self.vocab,
            'allow_oov': self.allow_oov,
            'encode_oov': self.encode_oov,
            'dtype': self.dtype
        }
        return config
    
    @classmethod
    def from_config(cls, config: dict) -> 'Feature':
        return cls(**config)
    
    @property 
    def name(self) -> str:
        return self.__class__.__name__
    
    @property 
    def output_dim(self) -> int:
        return 1 if not self.vocab else len(self.vocab)
    
    def _featurize_categorical(self, feature: str | int) -> np.ndarray:
        encoding = self.feature_to_onehot.get(feature, None)
        if encoding is not None:
            return encoding
        if not self.allow_oov:
            raise ValueError(
                f'{feature} could not be encoded, as it was not found in `vocab`. '
                'To allow OOV features, set `allow_oov` or `encode_oov` to True.'
            )
        oov_encoding = self.feature_to_onehot.get(self.oov_token, None)
        if oov_encoding is None:
            oov_encoding = np.zeros([self.output_dim], dtype=self.dtype) 
        return oov_encoding
    
    def _featurize_floating(self, value: float | list[float]) -> np.ndarray:
        if not isinstance(value, (int, float, bool)):
            raise ValueError(
                f'{self.name} produced a value of type {type(value)}. ' 
                'If it represents a categorical feature, please provide a `vocab` '
                'to the constructor. If if represents a floating point feature, '
                'please make sure its `call` method returns a list of values of '
                'type `float`, `int`, `bool` or `None`.'
            )
        if not math.isfinite(value):
            warn(
                f'Found value of {self.name} to be non-finite. '
                f'Value received: {value}. Converting it to a value of 0.'
            )
            value = 0.0
        return np.asarray([value], dtype=self.dtype)
    

@keras.saving.register_keras_serializable(package='molcraft')
class EdgeFeature(Feature):

    def __call__(self, mol: chem.Mol) -> np.ndarray:
        if not isinstance(mol, chem.Mol):
            raise ValueError(
                f'Input to {self.name} needs to be a `chem.Mol`, which '
                'implements two properties that should be iterated over '
                'to compute features: `atoms` and `bonds`.'
            )
        features = self.call(mol)
        if len(features) != int(mol.num_atoms**2):
            raise ValueError(
                f'The number of features computed by {self.name} does not '
                'match the number of node pairs in the `chem.Mol` object. '
                f'Make sure the list of items returned by {self.name}(input) '
                'correspond to node/atom pairs: '
                '[(0, 0), (0, 1), ..., (0, N), (1, 0), ... (N, N)], '
                'where N denotes the number of nodes/atoms.'
            )
        func = (
            self._featurize_categorical if self.vocab else 
            self._featurize_floating
        )
        return np.asarray([func(x) for x in features], dtype=self.dtype)
        

@keras.saving.register_keras_serializable(package='molcraft')
class Distance(EdgeFeature):
    
    def __init__(
        self, 
        max_distance: int = None, 
        allow_oov: int = True,
        encode_oov: bool = True, 
        **kwargs,
    ) -> None:
        vocab = kwargs.pop('vocab', None)
        if not vocab:
            if max_distance is None:
                max_distance = 20
            vocab = list(range(max_distance + 1))
        super().__init__(
            vocab=vocab, 
            allow_oov=allow_oov, 
            encode_oov=encode_oov, 
            **kwargs
        )

    def call(self, mol: chem.Mol) -> list[int]:
        return [int(x) for x in chem.get_distances(mol).reshape(-1)]
    

@keras.saving.register_keras_serializable(package='molcraft')
class AtomType(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.GetSymbol() for atom in mol.atoms]
    

@keras.saving.register_keras_serializable(package='molcraft')
class Degree(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.GetDegree() for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class TotalNumHs(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.GetTotalNumHs() for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class TotalValence(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.GetTotalValence() for atom in mol.atoms]
    

@keras.saving.register_keras_serializable(package='molcraft')
class AtomicWeight(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        pt = chem.get_periodic_table()
        return [pt.GetAtomicWeight(atom.GetSymbol()) for atom in mol.atoms]
    

@keras.saving.register_keras_serializable(package='molcraft')
class Hybridization(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [str(atom.GetHybridization()).lower() for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class CIPCode(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [
            atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else "None" 
            for atom in mol.atoms]
    

@keras.saving.register_keras_serializable(package='molcraft')
class IsChiralityPossible(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.HasProp("_ChiralityPossible") for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class FormalCharge(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.GetFormalCharge() for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class NumRadicalElectrons(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.GetNumRadicalElectrons() for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class IsAromatic(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return [atom.GetIsAromatic() for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class IsHetero(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return chem.hetero_atoms(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class IsHydrogenDonor(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return chem.hydrogen_donors(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class IsHydrogenAcceptor(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        return chem.hydrogen_acceptors(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class RingSize(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]: 
        def ring_size(atom):
            if not atom.IsInRing():
                return -1
            size = 3
            while not atom.IsInRingSize(size):
                size += 1 
            return size
        return [ring_size(atom) for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class IsInRing(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:         
        return [atom.IsInRing() for atom in mol.atoms]


@keras.saving.register_keras_serializable(package='molcraft')
class CrippenLogPContribution(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:      
        return chem.logp_contributions(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class CrippenMolarRefractivityContribution(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:      
        return chem.molar_refractivity_contribution(mol)
    

@keras.saving.register_keras_serializable(package='molcraft')
class TPSAContribution(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:      
        return chem.tpsa_contribution(mol)
    

@keras.saving.register_keras_serializable(package='molcraft')
class LabuteASAContribution(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:  
        return chem.asa_contribution(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class GasteigerCharge(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:  
        return chem.gasteiger_charges(mol)
    

@keras.saving.register_keras_serializable(package='molcraft')
class BondType(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:  
        return [str(bond.GetBondType()).lower() for bond in mol.bonds]


@keras.saving.register_keras_serializable(package='molcraft')
class Stereo(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:  
        return [
            str(bond.GetStereo()).replace('STEREO', '').capitalize() 
            for bond in mol.bonds
        ]
    

@keras.saving.register_keras_serializable(package='molcraft')
class IsConjugated(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:  
        return [bond.GetIsConjugated() for bond in mol.bonds]


@keras.saving.register_keras_serializable(package='molcraft')
class IsRotatable(Feature):
    def call(self, mol: chem.Mol) -> list[int, float, str]:  
        return chem.rotatable_bonds(mol)


default_vocabulary = {
    'AtomType': [
        '*', 'H', 'He', 'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne', 'Na', 
        'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar', 'K', 'Ca', 'Sc', 'Ti', 'V', 
        'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn', 'Ga', 'Ge', 'As', 'Se', 
        'Br', 'Kr', 'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 
        'Pd', 'Ag', 'Cd', 'In', 'Sn', 'Sb', 'Te', 'I', 'Xe', 'Cs', 'Ba', 
        'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy', 'Ho', 
        'Er', 'Tm', 'Yb', 'Lu', 'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 
        'Au', 'Hg', 'Tl', 'Pb', 'Bi', 'Po', 'At', 'Rn', 'Fr', 'Ra', 'Ac', 
        'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk', 'Cf', 'Es', 'Fm', 
        'Md', 'No', 'Lr', 'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 
        'Cn', 'Nh', 'Fl', 'Mc', 'Lv', 'Ts', 'Og'
    ],
    'Degree': [
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ],
    'TotalNumHs': [
        0, 1, 2, 3, 4
    ],
    'TotalValence': [
        0, 1, 2, 3, 4, 5, 6, 7, 8
    ],
    'Hybridization': [
        "s", "sp", "sp2", "sp3", "sp3d", "sp3d2", "unspecified"
    ],
    'CIPCode': [
        "R", "S", "None"
    ],
    'FormalCharge': [
        -3, -2, -1, 0, 1, 2, 3
    ],
    'NumRadicalElectrons': [
        0, 1, 2, 3, 4
    ],
    'RingSize': [
        -1, 3, 4, 5, 6, 7, 8
    ],
    'BondType': [
        "zero", "single", "double", "triple", "aromatic"
    ],
    'Stereo': [
        "E", "Z", "Any", "None"
    ],
}


def warn(message: str) -> None:
    warnings.warn(
        message=message,
        category=UserWarning,
        stacklevel=1
    )