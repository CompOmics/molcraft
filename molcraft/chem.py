import warnings
import collections
import numpy as np

from rdkit import Chem
from rdkit.Chem import Lipinski
from rdkit.Chem import rdDistGeom
from rdkit.Chem import rdDepictor
from rdkit.Chem import rdMolAlign
from rdkit.Chem import rdMolTransforms
from rdkit.Chem import rdPartialCharges
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdForceFieldHelpers
from rdkit.Chem import rdFingerprintGenerator


class Mol(Chem.Mol):
 
    @classmethod
    def from_encoding(cls, encoding: str, explicit_hs: bool = False, **kwargs) -> 'Mol':
        rdkit_mol = get_mol(encoding, **kwargs)
        if not rdkit_mol:
            return None
        if explicit_hs:
            rdkit_mol = Chem.AddHs(rdkit_mol) 
        rdkit_mol.__class__ = cls 
        return rdkit_mol

    @property
    def canonical_smiles(self) -> str:
        return Chem.MolToSmiles(self, canonical=True)
    
    @property
    def bonds(self) -> list['Bond']:
        if not hasattr(self, '_bonds'):
            self._bonds = get_bonds(self) 
        return self._bonds
    
    @property
    def atoms(self) -> list['Atom']:
        if not hasattr(self, '_atoms'):
            self._atoms = get_atoms(self)
        return self._atoms
    
    @property
    def num_conformers(self) -> int:
        return int(self.GetNumConformers())
    
    @property
    def num_atoms(self) -> int:
        return int(self.GetNumAtoms())
    
    @property 
    def num_bonds(self) -> int:
        return int(self.GetNumBonds())
    
    def get_atom(
        self, 
        atom: int | Chem.Atom
    ) -> 'Atom':
        if isinstance(atom, Chem.Atom):
            atom = atom.GetIdx()
        return Atom.cast(self.GetAtomWithIdx(int(atom)))
    
    def get_path_between_atoms(
        self, 
        atom_i: int | Chem.Atom, 
        atom_j: int | Chem.Atom
    ) -> tuple[int]:
        if isinstance(atom_i, Chem.Atom):
            atom_i = atom_i.GetIdx()
        if isinstance(atom_j, Chem.Atom):
            atom_j = atom_j.GetIdx()
        return Chem.rdmolops.GetShortestPath(
            self, int(atom_i), int(atom_j)
        )

    def get_bond_between_atoms(
        self, 
        atom_i: int | Chem.Atom, 
        atom_j: int | Chem.Atom,
    ) -> 'Bond':
        if isinstance(atom_i, Chem.Atom):
            atom_i = atom_i.GetIdx()
        if isinstance(atom_j, Chem.Atom):
            atom_j = atom_j.GetIdx()
        return Bond.cast(self.GetBondBetweenAtoms(int(atom_i), int(atom_j)))
        
    def adjacency(
        self, 
        fill: str = 'upper', 
        sparse: bool = True, 
        self_loops: bool = False, 
        dtype: str= 'int32', 
        cache: bool = True
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        if not hasattr(self, '_adjacency') or not cache:
            self._adjacency = get_adjacency_matrix(
                self, fill=fill, sparse=sparse, self_loops=self_loops, dtype=dtype
            )
        return self._adjacency

    def get_conformer(self, index: int = 0) -> 'Conformer':
        if self.num_conformers == 0:
            warn(
                'Molecule has no conformer. To embed conformer(s), invoke the `embed` method, '
                'and optionally followed by `minimize()` to perform force field minimization.'
            )
            return None
        return Conformer.cast(self.GetConformer(index))
    
    def get_conformers(self) -> list['Conformer']:
        if self.num_conformers == 0:
            warn(
                'Molecule has no conformers. To embed conformers, invoke the `embed` method, '
                'and optionally followed by `minimize()` to perform force field minimization.'
            )
            return []
        return [Conformer.cast(x) for x in self.GetConformers()]
     
    def __len__(self) -> int:
        return int(self.GetNumAtoms())
    
    def _repr_png_(self) -> None:
        return None
    
    def __repr__(self) -> str:
        return f'<{self.__class__.__name__} {self.canonical_smiles} at {hex(id(self))}>'


class Conformer(Chem.Conformer):

    @classmethod
    def cast(cls, obj: Chem.Conformer) -> 'Conformer':
        obj.__class__ = cls 
        return obj 

    @property 
    def index(self) -> int:
        return self.GetId()
    
    @property 
    def coordinates(self) -> np.ndarray:
        return self.GetPositions()
    
    @property
    def distances(self) -> np.ndarray:
        return Chem.rdmolops.Get3DDistanceMatrix(self.GetOwningMol())
    
    @property 
    def centroid(self) -> np.ndarray:
        return np.asarray(rdMolTransforms.ComputeCentroid(self))
    
    def adjacency(
        self, 
        fill: str = 'full',
        radius: float = None,
        sparse: bool = True, 
        self_loops: bool = False, 
        dtype: str = 'int32'
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        radius = radius or np.inf
        distances = self.distances
        if not self_loops:
            np.fill_diagonal(distances, np.inf)
        within_radius = distances < radius
        if fill == 'lower':
            within_radius = np.tril(within_radius, k=-1)
        elif fill == 'upper':
            within_radius = np.triu(within_radius, k=1)
        if sparse:
            edge_source, edge_target = np.where(within_radius)
            return edge_source.astype(dtype), edge_target.astype(dtype)
        return within_radius.astype(dtype)
        

class Atom(Chem.Atom):

    @classmethod
    def cast(cls, obj: Chem.Atom) -> 'Atom':
        obj.__class__ = cls 
        return obj 

    @property 
    def index(self) -> int:
        return int(self.GetIdx())
    
    @property
    def neighbors(self) -> list['Atom']:
        return [Atom.cast(neighbor) for neighbor in self.GetNeighbors()]
    
    def __repr__(self) -> str:
        return f'<Atom {self.GetSymbol()} at {hex(id(self))}>'


class Bond(Chem.Bond):

    @classmethod
    def cast(cls, obj: Chem.Bond) -> 'Bond':
        obj.__class__ = cls 
        return obj 
    
    @property 
    def index(self) -> int:
        return int(self.GetIdx())
    
    def __repr__(self) -> str:
        return f'<Bond {self.GetBondType().name} at {hex(id(self))}>'
    

def get_mol(
    encoding: str,
    strict: bool = True,
    assign_stereo_chemistry: bool = True,
) -> Chem.Mol:
    if isinstance(encoding, Chem.Mol):
        return encoding
    if encoding.startswith('InChI'):
        mol = Chem.MolFromInchi(encoding, sanitize=False)
    else:
        mol = Chem.MolFromSmiles(encoding, sanitize=False)
    if mol is not None:
        return sanitize_mol(mol, strict, assign_stereo_chemistry)
    raise ValueError(
        f"{encoding} is invalid; "
        f"make sure {encoding} is a valid SMILES or InChI string."
    )

def get_adjacency_matrix(
    mol: Chem.Mol,
    fill: str = 'full',
    sparse: bool = False,
    self_loops: bool = False,
    dtype: str = "int32",
) -> tuple[np.ndarray, np.ndarray]:
    adjacency: np.ndarray = Chem.GetAdjacencyMatrix(mol)
    if fill == 'lower':
        adjacency = np.tril(adjacency, k=-1)
    elif fill == 'upper':
        adjacency = np.triu(adjacency, k=1)
    if self_loops:
        adjacency += np.eye(adjacency.shape[0], dtype=adjacency.dtype)
    if not sparse:
        return adjacency.astype(dtype)
    edge_source, edge_target = np.where(adjacency)
    return edge_source.astype(dtype), edge_target.astype(dtype)

def sanitize_mol(
    mol: Chem.Mol,
    strict: bool = True,
    assign_stereo_chemistry: bool = True,
) -> Chem.Mol:
    flag = Chem.SanitizeMol(mol, catchErrors=True)
    if flag != Chem.SanitizeFlags.SANITIZE_NONE:
        if strict:
            return None
        # Sanitize mol, excluding the steps causing the error previously
        Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL^flag)
    if assign_stereo_chemistry:
        Chem.AssignStereochemistry(
            mol, cleanIt=True, force=True, flagPossibleStereoCenters=True)
    return mol

def get_atoms(mol: Mol) -> list[Atom]:
    return [
        Atom.cast(mol.GetAtomWithIdx(i)) 
        for i in range(mol.GetNumAtoms())
    ]

def get_bonds(mol: Mol) -> list[Bond]:
    return [
        Bond.cast(mol.GetBondWithIdx(int(i)))
        for i in range(mol.GetNumBonds())
    ]

def add_hs(mol: Mol) -> Mol:
    rdkit_mol = Chem.AddHs(mol)
    rdkit_mol.__class__ = mol.__class__
    return rdkit_mol

def remove_hs(mol: Mol) -> Mol:
    rdkit_mol = Chem.RemoveHs(mol)
    rdkit_mol.__class__ = mol.__class__
    return rdkit_mol

def get_distances(
    mol: Mol,
    fill: str = 'full',
    use_bond_order: bool = False, 
    use_atom_weights: bool = False
) -> np.ndarray:
    dist_matrix = Chem.rdmolops.GetDistanceMatrix(
        mol, useBO=use_bond_order, useAtomWts=use_atom_weights
    )
    # For disconnected nodes, a value of 1e8 is assigned to dist_matrix
    # Here we convert this large value to -1.
    # TODO: Add argument for filling disconnected node pairs.
    dist_matrix = np.where(
        dist_matrix >= 1e6, -1, dist_matrix
    )
    if fill == 'lower':
        return np.tril(dist_matrix, k=-1)
    elif fill == 'upper':
        return np.triu(dist_matrix, k=1)
    return dist_matrix

def get_shortest_paths( 
    mol: Mol, 
    radius: int, 
    self_loops: bool = False,
) -> list[list[int]]:
    paths = []
    for atom in mol.atoms:
        queue = collections.deque([(atom, [atom.index])])
        visited = set([atom.index])
        while queue:
            current_atom, path = queue.popleft()
            if len(path) > (radius + 1):
                continue
            if len(path) > 1 or self_loops:
                paths.append(path)
            for neighbor in current_atom.neighbors:
                if neighbor.index in visited:
                    continue
                visited.add(neighbor.index)
                queue.append((neighbor, path + [neighbor.index]))
    return paths

def get_periodic_table():
    return Chem.GetPeriodicTable()

def gasteiger_charges(mol: 'Mol') -> list[float]:
    rdPartialCharges.ComputeGasteigerCharges(mol)
    return [atom.GetDoubleProp("_GasteigerCharge") for atom in mol.atoms]

def logp_contributions(mol: 'Mol') -> list[float]:
    return [i[0] for i in rdMolDescriptors._CalcCrippenContribs(mol)]

def molar_refractivity_contribution(mol: 'Mol') -> list[float]:
    return [i[1] for i in rdMolDescriptors._CalcCrippenContribs(mol)]

def tpsa_contribution(mol: 'Mol') -> list[float]:
    return list(rdMolDescriptors._CalcTPSAContribs(mol))

def asa_contribution(mol: 'Mol') -> list[float]:
    return list(rdMolDescriptors._CalcLabuteASAContribs(mol)[0])

def hydrogen_acceptors(mol: 'Mol') -> list[bool]:
    h_acceptors = [i[0] for i in Lipinski._HAcceptors(mol)]
    return [atom.index in h_acceptors for atom in mol.atoms]

def hydrogen_donors(mol: 'Mol') -> list[bool]:
    h_donors = [i[0] for i in Lipinski._HDonors(mol)]
    return [atom.index in h_donors for atom in mol.atoms]

def hetero_atoms(mol: 'Mol') -> list[bool]:
    hetero_atoms = [i[0] for i in Lipinski._Heteroatoms(mol)]
    return [atom.index in hetero_atoms for atom in mol.atoms]

def rotatable_bonds(mol: 'Mol') -> list[bool]:
    rotatable_bonds = [set(x) for x in Lipinski._RotatableBonds(mol)]
    def is_rotatable(bond):
        atom_indices = {bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()}
        return atom_indices in rotatable_bonds
    return [is_rotatable(bond) for bond in mol.bonds]

def conformer_deviations(mol: Mol, fill: str = 'full') -> np.array:
    """Root mean squared deviation (RMSD) matrix"""
    num_confs = mol.num_conformers
    deviations = rdMolAlign.GetAllConformerBestRMS(mol)
    matrix = np.zeros((num_confs, num_confs))
    k = 0
    for i in range(num_confs):
        for j in range(i+1, num_confs):
            deviation = deviations[k]
            if fill == 'upper':
                matrix[i, j] = deviation 
            elif fill == 'lower':
                matrix[j, i] = deviation 
            else:
                matrix[i, j] = deviation
                matrix[j, i] = deviation
            k += 1
    return matrix

def conformer_energies(
    mol: Mol,
    method: str = 'UFF', 
) -> list[float]:
    if method == 'UFF':
        energies = _calc_uff_energies(mol)
    else:
        if method == 'MMFF':
            method += '94'
        variant = method
        energies = _calc_mmff_energies(mol, variant)
    return energies

def embed_conformers(
    mol: Mol, 
    num_conformers: int, 
    method: str = 'ETKDGv3',
    **kwargs
) -> None:
    available_embedding_methods = {
        'ETDG': rdDistGeom.ETDG(),
        'ETKDG': rdDistGeom.ETKDG(),
        'ETKDGv2': rdDistGeom.ETKDGv2(),
        'ETKDGv3': rdDistGeom.ETKDGv3(),
        'srETKDGv3': rdDistGeom.srETKDGv3(),
        'KDG': rdDistGeom.KDG()
    }
    mol = Mol(mol)
    embedding_method = available_embedding_methods.get(method)
    if embedding_method is None:
        raise ValueError(
            f'Could not find `method` {method!r}. Specify either of: '
            '`ETDG`, `ETKDG`, `ETKDGv2`, `ETKDGv3`, `srETKDGv3` or `KDG`.'
        )

    for key, value in kwargs.items():
        setattr(embedding_method, key, value)

    success = rdDistGeom.EmbedMultipleConfs(
        mol, numConfs=num_conformers, params=embedding_method
    )
    if not len(success):
        warn(
            f'Could not embed conformer(s) for {mol.canonical_smiles!r} using the '
            'speified method. Giving it another try with more permissive methods.'
        )
        max_attempts = (20 * mol.num_atoms) # increasing it from 10xN to 20xN
        for fallback_method in [method, 'ETDG', 'KDG']:
            fallback_embedding_method = available_embedding_methods[fallback_method]
            fallback_embedding_method.useRandomCoords = True
            fallback_embedding_method.maxAttempts = max_attempts
            success = rdDistGeom.EmbedMultipleConfs(
                mol, numConfs=num_conformers, params=fallback_embedding_method
            )
            if len(success):
                break
        else:
            raise RuntimeError(
                f'Could not embed conformer(s) for {mol.canonical_smiles!r}. '
            )
    return mol

def optimize_conformers(
    mol: Mol,
    method: str = 'UFF',
    max_iter: int = 200, 
    num_threads: bool = 1,
    ignore_interfragment_interactions: bool = True,
    vdw_threshold: float = 10.0,
):
    available_force_field_methods = [
        'MMFF', 'MMFF94', 'MMFF94s', 'UFF'
    ]
    if method not in available_force_field_methods:
        raise ValueError(
            f'Could not find `method` {method!r}. Specify either of: '
            '`UFF`, `MMFF`, `MMFF94` or `MMFF94s`.'
        )
    mol = Mol(mol)
    try:
        if method.startswith('MMFF'):
            variant = method 
            if variant == 'MMFF':
                variant += '94'
            _, _ = _mmff_optimize_conformers(
                mol, 
                num_threads=num_threads, 
                max_iter=max_iter, 
                variant=variant,
                ignore_interfragment_interactions=ignore_interfragment_interactions,
            )
        else:
            _, _ = _uff_optimize_conformers(
                mol,
                num_threads=num_threads,
                max_iter=max_iter,
                vdw_threshold=vdw_threshold,
                ignore_interfragment_interactions=ignore_interfragment_interactions,
            )
    except RuntimeError as e:
        warn(
            f'{method} force field minimization raised {e}. '
            '\nProceeding without force field minimization.'
        )
    return mol

def prune_conformers(
    mol: Mol,
    keep: int = 1,
    threshold: float = 0.0,
    energy_force_field: str = 'UFF',
):
    if mol.num_conformers == 0:
        warn(
            'Molecule has no conformers. To embed conformers, invoke the `embed` method, '
            'and optionally followed by `minimize()` to perform force field minimization.'
        )
        return mol
    
    threshold = threshold or 0.0
    deviations = conformer_deviations(mol)
    energies = conformer_energies(mol, method=energy_force_field)
    sorted_indices = np.argsort(energies)

    selected = [int(sorted_indices[0])]

    for target in sorted_indices[1:]:
        if len(selected) >= keep:
            break 
        if np.all(deviations[target, selected] >= threshold):
            selected.append(int(target))

    mol_copy = Mol(mol)
    mol_copy.RemoveAllConformers()
    for cid in selected:
        conformer = mol.get_conformer(cid)
        mol_copy.AddConformer(conformer, assignId=True)

    return mol_copy

def _uff_optimize_conformers(
    mol: Mol,
    num_threads: int = 1,
    max_iter: int = 200,
    vdw_threshold: float = 10.0,
    ignore_interfragment_interactions: bool = True,
    **kwargs,
) -> Mol:
    """Universal Force Field Minimization.
    """
    results = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(
        mol,
        numThreads=num_threads,
        maxIters=max_iter,
        vdwThresh=vdw_threshold,
        ignoreInterfragInteractions=ignore_interfragment_interactions,
    )
    energies = [r[1] for r in results]
    converged = [r[0] == 0 for r in results]
    return energies, converged 

def _mmff_optimize_conformers(
    mol: Mol,
    num_threads: int = 1,
    max_iter: int = 200,
    variant: str = 'MMFF94',
    ignore_interfragment_interactions: bool = True,
    **kwargs,
) -> Mol:
    """Merck Molecular Force Field Minimization.
    """
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        raise ValueError("Cannot minimize molecule using MMFF.")
    rdForceFieldHelpers.MMFFSanitizeMolecule(mol)
    results = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
        mol,
        num_threads=num_threads,
        maxIters=max_iter,
        mmffVariant=variant,
        ignoreInterfragInteractions=ignore_interfragment_interactions,
    )
    energies = [r[1] for r in results]
    converged = [r[0] == 0 for r in results]
    return energies, converged 

def _calc_uff_energies(
    mol: Mol,
) -> list[float]:
    energies = []
    for i in range(mol.num_conformers):
        try:
            force_field = rdForceFieldHelpers.UFFGetMoleculeForceField(mol, confId=i)
            energies.append(force_field.CalcEnergy())
        except Exception:
            energies.append(float('nan'))
    return energies
    
def _calc_mmff_energies(
    mol: Mol,
    variant: str = 'MMFF94',
) -> list[float]:
    energies = []
    if not rdForceFieldHelpers.MMFFHasAllMoleculeParams(mol):
        raise ValueError("Cannot compute MMFF energies for this molecule.")
    props = rdForceFieldHelpers.MMFFGetMoleculeProperties(mol, mmffVariant=variant)
    for i in range(mol.num_conformers):
        try:
            force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(mol, props, confId=i)
            energies.append(force_field.CalcEnergy())
        except Exception:
            energies.append(float('nan'))
    return energies

def unpack_conformers(mol: Mol) -> list[Mol]:
    mols = []
    for conf in mol.get_conformers():
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        new_mol.AddConformer(conf, assignId=True)
        new_mol.__class__ = mol.__class__
        mols.append(new_mol)
    return mols

_fingerprint_types = {
    'rdkit': rdFingerprintGenerator.GetRDKitFPGenerator,
    'morgan': rdFingerprintGenerator.GetMorganGenerator,
    'topological_torsion': rdFingerprintGenerator.GetTopologicalTorsionGenerator,
    'atom_pair': rdFingerprintGenerator.GetAtomPairGenerator,
}

def _get_fingerprint(
    mol: Mol,
    fp_type: str = 'morgan',
    binary: bool = True,
    dtype: str = 'float32',
    **kwargs,
) -> np.ndarray:
    fingerprint: rdFingerprintGenerator.FingerprintGenerator64 = (
        _fingerprint_types[fp_type](**kwargs)
    )
    if not isinstance(mol, Mol):
        mol = Mol.from_encoding(mol)
    if binary:
        fp: np.ndarray = fingerprint.GetFingerprintAsNumPy(mol)
    else:
        fp: np.ndarray = fingerprint.GetCountFingerprintAsNumPy(mol)
    return fp.astype(dtype)

def _rdkit_fingerprint(
    mol: Chem.Mol, 
    size: int = 2048, 
    *,
    min_path: int = 1, 
    max_path: int = 7, 
    binary: bool = True,
    dtype: str = 'float32',
) -> np.ndarray:
    fp_param = {'fpSize': size, 'minPath': min_path, 'maxPath': max_path}
    return _get_fingerprint(mol, 'rdkit', binary, dtype, **fp_param)

def _morgan_fingerprint(
    mol: Chem.Mol, 
    size: int = 2048, 
    *,
    radius: int = 3, 
    binary: bool = True,
    dtype: str = 'float32',
) -> np.ndarray:
    fp_param = {'radius': radius, 'fpSize': size}
    return _get_fingerprint(mol, 'morgan', binary, dtype, **fp_param)

def _topological_torsion_fingerprint(
    mol: Chem.Mol, 
    size: int = 2048, 
    *,
    binary: bool = True,
    dtype: str = 'float32',
) -> np.ndarray:
    fp_param = {'fpSize': size}
    return _get_fingerprint(mol, 'topological_torsion', binary, dtype, **fp_param)

def _atom_pair_fingerprint(
    mol: Chem.Mol, 
    size: int = 2048, 
    *,
    binary: bool = True,
    dtype: str = 'float32',
) -> np.ndarray:
    fp_param = {'fpSize': size}
    return _get_fingerprint(mol, 'atom_pair', binary, dtype, **fp_param)

def warn(message: str) -> None:
    warnings.warn(
        message=message,
        category=UserWarning,
        stacklevel=1,
    )
