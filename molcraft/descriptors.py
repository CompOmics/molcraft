import keras
import numpy as np
from rdkit.Chem import rdMolDescriptors

from molcraft import chem
from molcraft import features


@keras.saving.register_keras_serializable(package='molcraft')
class Descriptor(features.Feature):

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
        if not isinstance(descriptor, (tuple, list, np.ndarray)):
            descriptor = [descriptor]
        
        descriptors = []
        for value in descriptor:
            descriptors.append(func(value))
        return np.concatenate(descriptors)
    

@keras.saving.register_keras_serializable(package='molcraft')
class MolWeight(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcExactMolWt(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class TPSA(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcTPSA(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class CrippenLogP(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
    

@keras.saving.register_keras_serializable(package='molcraft')
class CrippenMolarRefractivity(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcCrippenDescriptors(mol)[1]
    

@keras.saving.register_keras_serializable(package='molcraft')
class NumHeavyAtoms(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumHeavyAtoms(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class NumHeteroAtoms(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumHeteroatoms(mol) 
    
    
@keras.saving.register_keras_serializable(package='molcraft')
class NumHydrogenDonors(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumHBD(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class NumHydrogenAcceptors(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumHBA(mol) 
    

@keras.saving.register_keras_serializable(package='molcraft')
class NumRotatableBonds(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumRotatableBonds(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class NumRings(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumRings(mol) 

