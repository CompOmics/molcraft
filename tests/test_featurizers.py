import unittest 

from molcraft import features
from molcraft import featurizers
from molcraft import conformers


class TestFeaturizer(unittest.TestCase):

    def setUp(self):

        self._smiles_no_atom = [''] 
        self._smiles_single_atom = ['C']
        self._smiles_single_hs_atom = ['[H]']
        self._smiles_two_disconnected_singles = ['C.O']
        self._smiles_two_disconnected_doubles = ['CC.CO']
        self._smiles_one_molecule = [
            'C(C(=O)O)N'
        ]
        self._smiles_two_molecules = [
            'O=C([C@H](CC1=CNC=N1)N)O',
            'C(C(=O)O)N'
        ]
        self._smiles_single_double = [
            'C',
            'CO'
        ]

    def test_mol_featurizer(self):
            
        featurizer = featurizers.MolFeaturizer(
            atom_features=[
                features.AtomType({'C', 'N', 'O', 'H'}),
                features.TotalNumHs({0, 1, 2, 3, 4})
            ],
            bond_features=[
                features.BondType({'single', 'double', 'aromatic'}),
                features.IsRotatable(),
            ],
            molecule_features='auto',
            super_atom=True,
            radius=1, 
            self_loops=False,
            include_hs=False, 
        ) 

        node_dim = 9
        edge_dim = 4

        smiles = self._smiles_one_molecule
        num_nodes = (5 + 1)
        num_edges = (8 + 5 * 2)
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.context['feature'].shape, (1, 10))
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.edge['feature'].shape, (num_edges, edge_dim))
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')

        smiles = self._smiles_two_molecules
        num_nodes = (5 + 1) + (11 + 1)
        num_edges = (8 + 5 * 2) + (22 + 11 * 2)
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.context['feature'].shape, (2, 10))
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.edge['feature'].shape, (num_edges, edge_dim))
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')
        
        smiles = self._smiles_no_atom 
        num_nodes = (0 + 1)
        num_edges = (0 + 0 * 2)
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.context['feature'].shape, (1, 10))
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.edge['feature'].shape, (num_edges, edge_dim))
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')

        smiles = self._smiles_single_atom 
        num_nodes = (1 + 1)
        num_edges = (0 + 1 * 2)
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.context['feature'].shape, (1, 10))
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.edge['feature'].shape, (num_edges, edge_dim))
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')

        smiles = self._smiles_single_hs_atom 
        num_nodes = (1 + 1)
        num_edges = (0 + 1 * 2)
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.context['feature'].shape, (1, 10))
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.edge['feature'].shape, (num_edges, edge_dim))
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')


    def test_mol_featurizer3d(self):
        
        num_conformers = 5
        featurizer = featurizers.MolFeaturizer3D(
            atom_features=[
                features.AtomType({'C', 'N', 'O', 'H'}, encode_oov=True),
                features.TotalNumHs({0, 1, 2, 3, 4})
            ],
            bond_features=[
                features.Distance(max_distance=20)
            ],
            conformer_generator=conformers.ConformerEmbedder(
                method='ETKDGv3',
                num_conformers=num_conformers,
            ),
            super_atom=True,
            radius=5.0, 
            self_loops=False,
            include_hs=False, 
        ) 

        node_dim = 10
        edge_dim = 22

        smiles = self._smiles_one_molecule
        num_nodes = (5 + 1) * num_conformers
        num_edges = (8 + 5 * 2) * num_conformers
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.node['coordinate'].shape, (num_nodes, 3))
            self.assertGreaterEqual(graph.edge['feature'].shape[0], num_edges)
            self.assertEqual(graph.edge['feature'].shape[-1], edge_dim)
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')

        smiles = self._smiles_two_molecules
        num_nodes = ((5 + 1) + (11 + 1)) * num_conformers
        num_edges = ((8 + 5 * 2) + (22 + 11 * 2)) * num_conformers
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.node['coordinate'].shape, (num_nodes, 3))
            self.assertGreaterEqual(graph.edge['feature'].shape[0], num_edges)
            self.assertEqual(graph.edge['feature'].shape[-1], edge_dim)
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')

        smiles = self._smiles_single_atom 
        num_nodes = (1 + 1) * num_conformers
        num_edges = (0 + 1 * 2) * num_conformers
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.edge['feature'].shape, (num_edges, edge_dim))
            self.assertEqual(graph.node['coordinate'].shape, (num_nodes, 3))
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')

        smiles = self._smiles_single_hs_atom 
        num_nodes = (1 + 1) * num_conformers
        num_edges = (0 + 1 * 2) * num_conformers
        with self.subTest(smiles=smiles):
            graph = featurizer(smiles)
            self.assertEqual(graph.node['feature'].shape, (num_nodes, node_dim))
            self.assertEqual(graph.edge['feature'].shape, (num_edges, edge_dim))
            self.assertEqual(graph.node['coordinate'].shape, (num_nodes, 3))
            self.assertEqual(graph.node['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['feature'].dtype.name, 'float32')
            self.assertEqual(graph.edge['source'].dtype.name, 'int32')
            self.assertEqual(graph.edge['target'].dtype.name, 'int32')



if __name__ == '__main__':
    unittest.main()