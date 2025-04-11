import keras
import numpy as np
from rdkit.Chem import Descriptors

from molcraft import chem
from molcraft import features


@keras.saving.register_keras_serializable(package='molcraft')
class Descriptor(features.Feature):
    def __init__(self, scale: float | None = None, **kwargs):
        super().__init__(**kwargs)
        self.scale = scale 

    def __call__(self, mol: chem.Mol) -> np.ndarray:
        if not isinstance(mol, chem.Mol):
            raise ValueError(
                f'Input to {self.name} needs to be a `chem.Mol`, which '
                'implements two properties that should be iterated over '
                'to compute features: `atoms` and `bonds`.'
            )
        descriptor = self.call(mol)
        func = (
            self._featurize_categorical if self.vocab else 
            self._featurize_floating
        )
        scale_value = self.scale and not self.vocab
        if not isinstance(descriptor, (tuple, list, np.ndarray)):
            descriptor = [descriptor]
        
        descriptors = []
        for value in descriptor:
            if scale_value:
                value /= self.scale 
            descriptors.append(func(value))
        return np.concatenate(descriptors)
    
    def get_config(self):
        config = super().get_config()
        config['scale'] = self.scale 
        return config 
    

@keras.saving.register_keras_serializable(package='molcraft')
class MolWeight(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.MolWt(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class MolTPSA(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.TPSA(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class MolLogP(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.MolLogP(mol)
    

@keras.saving.register_keras_serializable(package='molcraft')
class NumHeavyAtoms(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.HeavyAtomCount(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class NumHydrogenDonors(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.NumHDonors(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class NumHydrogenAcceptors(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.NumHAcceptors(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class NumRotatableBonds(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.NumRotatableBonds(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class NumRings(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return Descriptors.RingCount(mol) 

