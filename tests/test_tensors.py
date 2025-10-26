import unittest 

import keras
import tensorflow as tf

from molcraft import tensors 


class TestGraphTensor(unittest.TestCase):

    def setUp(self):
        self.flat_tensors = [
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

        self.unflat_tensors = [
            # Graph with two subgraphs
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3], dtype='int32')
                },
                node={
                    'feature': tf.ragged.constant([[[1., 2.], [3., 4.]], [[5., 6.], [6., 7.], [8., 9.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32'),
                    'weight': tf.ragged.constant([[0.50, 1.00], [2.00, 0.25, 0.75]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                },
                edge={
                    'source': tf.ragged.constant([[0, 1], [1, 2, 2, 1, 0]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'target': tf.ragged.constant([[1, 0], [0, 1, 2, 2, 1]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'feature': tf.ragged.constant([[[1.], [2.]], [[3.], [4.], [5.], [6.], [7.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                }
            ),
            # Graph with two subgraphs, none which has edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3], dtype='int32')
                },
                node={
                    'feature': tf.ragged.constant([[[1., 2.], [3., 4.]], [[5., 6.], [6., 7.], [8., 9.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32'),
                    'weight': tf.ragged.constant([[0.50, 1.00], [2.00, 0.25, 0.75]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                },
                edge={
                    'source': tf.ragged.constant([[], []], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'target': tf.ragged.constant([[], []], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'feature': tf.RaggedTensor.from_value_rowids(keras.ops.zeros([0, 1], dtype='float32'), keras.ops.zeros([0], dtype='int32'), 2),
                }
            ),
            # Graph with two subgraphs, none of which has nodes or edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([0, 0], dtype='int32')
                },
                node={
                    'feature': tf.RaggedTensor.from_value_rowids(keras.ops.zeros([0, 2], dtype='float32'), keras.ops.zeros([0], dtype='int32'), 2),
                    'weight': tf.RaggedTensor.from_value_rowids(keras.ops.zeros([0], dtype='float32'), keras.ops.zeros([0], dtype='int32'), 2)
                },
                edge={
                    'source': tf.ragged.constant([[], []], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'target': tf.ragged.constant([[], []], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'feature': tf.RaggedTensor.from_value_rowids(keras.ops.zeros([0, 1], dtype='float32'), keras.ops.zeros([0], dtype='int32'), 2)
                }
            ),
            # Graph with three subgraphs where the second subgraph's first node has no edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3, 2], dtype='int32')
                },
                node={
                    'feature': tf.ragged.constant([[[1., 2.], [3., 4.]], [[5., 6.], [6., 7.], [8., 9.]], [[10., 11.], [12., 13.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32'),
                    'weight': tf.ragged.constant([[0.50, 1.00], [2.00, 0.25, 0.75], [0.25, 0.25]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                },
                edge={
                    'source': tf.ragged.constant([[0, 1], [2, 1, 2], [0, 1, 1]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'target': tf.ragged.constant([[1, 0], [1, 2, 2], [1, 0, 1]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'feature': tf.ragged.constant([[[1.], [2.]], [[3.], [4.], [5.]], [[6.], [7.], [8.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                }
            ),
            # Graph with three subgraphs where the first subgraph's nodes have no edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3, 2], dtype='int32')
                },
                node={
                    'feature': tf.ragged.constant([[[1., 2.], [3., 4.]], [[5., 6.], [6., 7.], [8., 9.]], [[10., 11.], [12., 13.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32'),
                    'weight': tf.ragged.constant([[0.50, 1.00], [2.00, 0.25, 0.75], [0.25, 0.25]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                },
                edge={
                    'source': tf.ragged.constant([[], [2, 1, 2], [0, 1, 1]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'target': tf.ragged.constant([[], [1, 2, 2], [1, 0, 1]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'feature': tf.ragged.constant([[], [[3.], [4.], [5.]], [[6.], [7.], [8.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                }
            ),
            # Graph with three subgraphs where the last subgraph's nodes have no edges
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3, 3], dtype='int32')
                },
                node={
                    'feature': tf.ragged.constant([[[1., 2.], [3., 4.]], [[5., 6.], [6., 7.], [8., 9.]], [[10., 11.], [12., 13.], [14., 15.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32'),
                    'weight': tf.ragged.constant([[0.50, 1.00], [2.00, 0.25, 0.75], [0.25, 0.25, 0.5]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                },
                edge={
                    'source': tf.ragged.constant([[0, 1], [2, 1, 2], [0, 1]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'target': tf.ragged.constant([[1, 0], [1, 2, 2], [1, 0]], dtype='int32', ragged_rank=1, row_splits_dtype='int32'),
                    'feature': tf.ragged.constant([[[1.], [2.]], [[3.], [4.], [5.]], [[6.], [7.]]], dtype='float32', ragged_rank=1, row_splits_dtype='int32')
                }
            ),
        ]

    def test_unflatten(self):
        for i, (unflat_tensor, flat_tensor) in enumerate(zip(self.unflat_tensors, self.flat_tensors)):
            with self.subTest(id=i):
                self.assertTrue(bool(unflat_tensor == flat_tensor.unflatten()))
            

    def test_flatten(self):
        for i, (unflat_tensor, flat_tensor) in enumerate(zip(self.unflat_tensors, self.flat_tensors)):
            with self.subTest(i=i):
                self.assertTrue(bool(unflat_tensor.flatten() == flat_tensor))

    def test_tf_dataset(self):
        for i, tensor in enumerate(self.flat_tensors):
            with self.subTest(i=i, flat=True):
                ds = tf.data.Dataset.from_tensor_slices(tensor)
                for x in ds.batch(2).unbatch().batch(2).take(1):
                    pass
                self.assertTrue(x == tensor[:2])

        for i, tensor in enumerate(self.unflat_tensors):
            with self.subTest(i=i, flat=False):
                ds = tf.data.Dataset.from_tensor_slices(tensor)
                for x in ds.batch(2).unbatch().batch(2).take(1):
                    pass
                self.assertTrue(x == tensor[:2])


if __name__ == '__main__':
    unittest.main()