import unittest 

import keras

from molcraft import tensors 
from molcraft import layers 


class TestLayer(unittest.TestCase):

    def setUp(self):

        self.tensors = [
            # Graph with two subgraphs
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3], dtype='int32')
                },
                node={
                    'feature': keras.ops.array([[1., 2.], [3., 4.], [5., 6.], [6., 7.], [8., 9.]], dtype='float32'),
                    'weight': keras.ops.array([0.50, 1.00, 2.00, 0.25, 0.75], dtype='float32')
                },
                edge={
                    'source': keras.ops.array([0, 1, 3, 4, 4, 3, 2], dtype='int32'),
                    'target': keras.ops.array([1, 0, 2, 3, 4, 4, 3], dtype='int32'),
                    'feature': keras.ops.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.]], dtype='float32')
                }
            ),
            # Graph with two subgraphs, none which has edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3], dtype='int32')
                },
                node={
                    'feature': keras.ops.array([[1., 2.], [3., 4.], [5., 6.], [6., 7.], [8., 9.]], dtype='float32'),
                    'weight': keras.ops.array([0.50, 1.00, 2.00, 0.25, 0.75], dtype='float32')
                },
                edge={
                    'source': keras.ops.zeros([0], dtype='int32'),
                    'target': keras.ops.zeros([0], dtype='int32'),
                    'feature': keras.ops.zeros([0, 1], dtype='float32')
                }
            ),
            # Graph with two subgraphs, none of which has nodes or edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([0, 0], dtype='int32')
                },
                node={
                    'feature': keras.ops.zeros([0, 2], dtype='float32'),
                    'weight': keras.ops.zeros([0], dtype='float32')
                },
                edge={
                    'source': keras.ops.zeros([0], dtype='int32'),
                    'target': keras.ops.zeros([0], dtype='int32'),
                    'feature': keras.ops.zeros([0, 1], dtype='float32')
                }
            ),
            # Graph with three subgraphs where the second subgraph's first node has no edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3, 2], dtype='int32')
                },
                node={
                    'feature': keras.ops.array([[1., 2.], [3., 4.], [5., 6.], [6., 7.], [8., 9.], [10., 11.], [12., 13.]], dtype='float32'),
                    'weight': keras.ops.array([0.50, 1.00, 2.00, 0.25, 0.75, 0.25, 0.25], dtype='float32')
                },
                edge={
                    'source': keras.ops.array([0, 1, 4, 3, 4, 5, 6, 6], dtype='int32'),
                    'target': keras.ops.array([1, 0, 3, 4, 4, 6, 5, 6], dtype='int32'),
                    'feature': keras.ops.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.], [8.]], dtype='float32')
                }
            ),
            # Graph with three subgraphs where the first subgraph's nodes have no edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3, 2], dtype='int32')
                },
                node={
                    'feature': keras.ops.array([[1., 2.], [3., 4.], [5., 6.], [6., 7.], [8., 9.], [10., 11.], [12., 13.]], dtype='float32'),
                    'weight': keras.ops.array([0.50, 1.00, 2.00, 0.25, 0.75, 0.25, 0.25], dtype='float32')
                },
                edge={
                    'source': keras.ops.array([4, 3, 4, 5, 6, 6], dtype='int32'),
                    'target': keras.ops.array([3, 4, 4, 6, 5, 6], dtype='int32'),
                    'feature': keras.ops.array([[3.], [4.], [5.], [6.], [7.], [8.]], dtype='float32')
                }
            ),
            # Graph with three subgraphs where the last subgraph's nodes have no edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3, 3], dtype='int32')
                },
                node={
                    'feature': keras.ops.array([[1., 2.], [3., 4.], [5., 6.], [6., 7.], [8., 9.], [10., 11.], [12., 13.], [14., 15.]], dtype='float32'),
                    'weight': keras.ops.array([0.50, 1.00, 2.00, 0.25, 0.75, 0.25, 0.25, 0.5], dtype='float32')
                },
                edge={
                    'source': keras.ops.array([0, 1, 4, 3, 4, 5, 6], dtype='int32'),
                    'target': keras.ops.array([1, 0, 3, 4, 4, 6, 5], dtype='int32'),
                    'feature': keras.ops.array([[1.], [2.], [3.], [4.], [5.], [6.], [7.]], dtype='float32')
                }
            ),
        ]

    def test_graph_conv(self):
        for i, tensor in enumerate(self.tensors):
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.GraphConv(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == tensor.edge['feature'].shape[-1])

    def test_gi_conv(self):
        for i, tensor in enumerate(self.tensors):
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.GIConv(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == tensor.node['feature'].shape[-1])

    def test_ga_conv(self):
        for i, tensor in enumerate(self.tensors):
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.GAConv(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == 128)

    def test_mp_conv(self):
        for i, tensor in enumerate(self.tensors):
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.MPConv(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == tensor.edge['feature'].shape[-1])

    def test_gt_conv(self):
        for i, tensor in enumerate(self.tensors):
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.GTConv(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == tensor.edge['feature'].shape[-1])
                        
    def test_mp_conv3d(self):
        for i, tensor in enumerate(self.tensors):
            tensor = tensor.update(
                {
                    'node': {
                        'coordinate': keras.random.uniform(
                            shape=(tensor.node['feature'].shape[0], 3), minval=-10, maxval=10)
                    }
                }
            )
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.MPConv3D(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            self.assertTrue(output.node['coordinate'].shape[-1] == tensor.node['coordinate'].shape[-1])
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == tensor.edge['feature'].shape[-1])

    def test_eg_conv3d(self):
        for i, tensor in enumerate(self.tensors):
            tensor = tensor.update(
                {
                    'node': {
                        'coordinate': keras.random.uniform(
                            shape=(tensor.node['feature'].shape[0], 3), minval=-10, maxval=10)
                    }
                }
            )
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.EGConv3D(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            self.assertTrue(output.node['coordinate'].shape[-1] == tensor.node['coordinate'].shape[-1])
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == tensor.edge['feature'].shape[-1])

    def test_gt_conv3d(self):
        for i, tensor in enumerate(self.tensors):
            tensor = tensor.update(
                {
                    'node': {
                        'coordinate': keras.random.uniform(
                            shape=(tensor.node['feature'].shape[0], 3), minval=-10, maxval=10)
                    }
                }
            )
            for drop_edge_feature in [False, True]:
                if drop_edge_feature:
                    tensor = tensor.update(
                        {
                            'edge': {
                                'feature': None
                            }
                        }
                    )
                for skip_connect in [True, False, 'weighted']:
                    for normalize in [True, False, 'batch']:
                        with self.subTest(i=i, skip_connect=skip_connect, normalize=normalize, flat=True):
                            output = layers.GTConv3D(128, skip_connect=skip_connect, normalize=normalize)(tensor)
                            self.assertTrue(output.node['feature'].shape[-1] == 128)
                            self.assertTrue(output.node['coordinate'].shape[-1] == tensor.node['coordinate'].shape[-1])
                            if not drop_edge_feature:
                                self.assertTrue(output.edge['feature'].shape[-1] == tensor.edge['feature'].shape[-1])


if __name__ == '__main__':
    unittest.main()