import warnings
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
                f'Input to {self.name} must be a `chem.Mol` object.'
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
class Descriptor3D(Descriptor):
    
    def __call__(self, mol: chem.Mol) -> np.ndarray:
        if not isinstance(mol, chem.Mol):
            raise ValueError(
                f'Input to {self.name} must be a `chem.Mol` object.'
            ) 
        if mol.num_conformers == 0:
            raise ValueError(
                f'The inputted `chem.Mol` to {self.name} must embed a conformer. '
                f'It is recommended that {self.name} is used as a molecule feature '
                'for `MolGraphFeaturizer3D`, which by default embeds a conformer.'
            )
        return super().__call__(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class MolWeight(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcExactMolWt(mol) 


@keras.saving.register_keras_serializable(package='molcraft')
class TotalPolarSurfaceArea(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcTPSA(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class LogP(Descriptor):
    """Crippen logP."""
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
    

@keras.saving.register_keras_serializable(package='molcraft')
class MolarRefractivity(Descriptor):
    """Crippen molar refractivity."""
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcCrippenDescriptors(mol)[1]
    

@keras.saving.register_keras_serializable(package='molcraft')
class NumAtoms(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumAtoms(mol)
    

@keras.saving.register_keras_serializable(package='molcraft')
class NumHeavyAtoms(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumHeavyAtoms(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class NumHeteroatoms(Descriptor):
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


@keras.saving.register_keras_serializable(package='molcraft')
class NumAromaticRings(Descriptor):
    def call(self, mol: chem.Mol) -> np.ndarray:
        return rdMolDescriptors.CalcNumAromaticRings(mol)


@keras.saving.register_keras_serializable(package='molcraft')
class AtomCount(Descriptor):

    def __init__(self, atom_type: str, **kwargs):
        super().__init__(**kwargs)
        self.atom_type = atom_type

    def call(self, mol: chem.Mol) -> np.ndarray:
        count = 0
        for atom in mol.atoms:
            if atom.GetSymbol() == self.atom_type:
                count += 1
        return count
    
    def get_config(self) -> dict:
        config = super().get_config()
        config['atom_type'] = self.atom_type
        return config
    

@keras.saving.register_keras_serializable(package='molcraft')
class ForceFieldEnergy(Descriptor3D):
    """Universal Force Field (UFF) Energy."""
    def call(self, mol: chem.Mol) -> np.ndarray:
        return chem.conformer_energies(mol, method="UFF")
        