import unittest 
import tempfile
import shutil
import keras
import numpy as np

from molcraft import tensors 
from molcraft import layers 
from molcraft import models


class TestModel(unittest.TestCase):

    def setUp(self):

        self.tensors = [
            # Graph with two subgraphs
            tensors.GraphTensor(
                context={
                    'size': keras.ops.array([2, 3], dtype='int32'),
                    'label': keras.ops.array([1., 2.], dtype='float32'),
                    'weight': keras.ops.array([0.5, 1.25], dtype='float32'),
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
                    'size': keras.ops.array([2, 3], dtype='int32'),
                    'label': keras.ops.array([1., 2.], dtype='float32'),
                    'weight': keras.ops.array([0.5, 1.25], dtype='float32'),
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
                    'size': keras.ops.array([0, 0], dtype='int32'),
                    'label': keras.ops.array([1., 2.], dtype='float32'),
                    'weight': keras.ops.array([0.5, 1.25], dtype='float32'),
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
                    'size': keras.ops.array([2, 3, 2], dtype='int32'),
                    'label': keras.ops.array([1., 2., 3.], dtype='float32'),
                    'weight': keras.ops.array([0.5, 1.25, 0.75], dtype='float32'),
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
                    'size': keras.ops.array([2, 3, 2], dtype='int32'),
                    'label': keras.ops.array([1., 2., 3.], dtype='float32'),
                    'weight': keras.ops.array([0.5, 1.25, 0.75], dtype='float32'),
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
                    'size': keras.ops.array([2, 3, 3], dtype='int32'),
                    'label': keras.ops.array([1., 2., 3.], dtype='float32'),
                    'weight': keras.ops.array([0.5, 1.25, 0.75], dtype='float32'),
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

    def test_functional_model(self):

        def get_model(tensor):
            inputs = layers.Input(tensor.spec)
            x = layers.NodeEmbedding(32)(inputs)
            x = layers.EdgeEmbedding(32)(x)
            x = layers.GTConv(32)(x)
            x = layers.GTConv(32)(x)
            x = layers.Readout('sum')(x)
            outputs = keras.layers.Dense(1)(x)
            return models.GraphModel(inputs, outputs)
        
        for i, tensor in enumerate(self.tensors):
            with self.subTest(i=i, functional=True):
                model = get_model(tensor)
                output = model(tensor)
                self.assertTrue(output.shape[0] == tensor.context['label'].shape[0])
                model.compile('adam', 'mse', metrics=[keras.metrics.MeanAbsoluteError()])
                metrics = model.evaluate(tensor, verbose=0)
                self.assertTrue(isinstance(metrics, list))
                del model

    def test_saved_model(self):

        def get_model(tensor):
            inputs = layers.Input(tensor.spec)
            x = layers.NodeEmbedding(32)(inputs)
            x = layers.EdgeEmbedding(32)(x)
            x = layers.GTConv(32)(x)
            x = layers.GTConv(32)(x)
            x = layers.Readout('sum')(x)
            outputs = keras.layers.Dense(1)(x)
            return models.GraphModel(inputs, outputs)

        @keras.saving.register_keras_serializable()
        class Model(models.GraphModel):
            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                self.e1 = layers.NodeEmbedding(32)
                self.e2 = layers.EdgeEmbedding(32)
                self.c1 = layers.GTConv(32)
                self.c2 = layers.GTConv(32)
                self.r = layers.Readout('sum')
                self.d = keras.layers.Dense(1)
            def propagate(self, tensor):
                return self.d(self.r(self.c2(self.c1(self.e2(self.e1(tensor))))))
            
        example = self.tensors[-1]

        tmp_dir = tempfile.mkdtemp()
        tmp_file = tmp_dir + '/model.keras'
        with self.subTest(functional_model=True):
            model = get_model(example)
            model.compile('adam', 'mse')
            model.fit(example, verbose=0)
            model.save(tmp_file)
            loaded_model = models.load_model(tmp_file)
            pred_1 = model.predict(example, verbose=0)
            pred_2 = loaded_model.predict(example, verbose=0)
            test_preds = np.all(pred_1.round(4) == pred_2.round(4))
            self.assertTrue(test_preds)
            test_vars = np.all([np.all((v1 == v2).numpy()) for (v1, v2) in zip(model.variables, loaded_model.variables)])
            self.assertTrue(test_vars)
        shutil.rmtree(tmp_dir)

        tmp_dir = tempfile.mkdtemp()
        tmp_file = tmp_dir + '/model.keras'
        with self.subTest(functional_model=False):
            model = Model()
            model.compile('adam', 'mse')
            model.fit(example, verbose=0)
            model.save(tmp_file)
            loaded_model = models.load_model(tmp_file)
            pred_1 = model.predict(example, verbose=0)
            pred_2 = loaded_model.predict(example, verbose=0)
            test_preds = np.all(pred_1.round(4) == pred_2.round(4))
            self.assertTrue(test_preds)
            test_vars = np.all([np.all((v1 == v2).numpy()) for (v1, v2) in zip(model.variables, loaded_model.variables)])
            self.assertTrue(test_vars)
        shutil.rmtree(tmp_dir)

    def test_subclassed_model(self):

        def get_model(tensor):
            class Model(models.GraphModel):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.e1 = layers.NodeEmbedding(32)
                    self.e2 = layers.EdgeEmbedding(32)
                    self.c1 = layers.GTConv(32)
                    self.c2 = layers.GTConv(32)
                    self.r = layers.Readout('sum')
                    self.d = keras.layers.Dense(1)
                def propagate(self, tensor):
                    return self.d(self.r(self.c2(self.c1(self.e2(self.e1(tensor))))))
            return Model()
                
        for i, tensor in enumerate(self.tensors):
            with self.subTest(i=i, functional=True):
                model = get_model(tensor)
                output = model(tensor)
                self.assertTrue(output.shape[0] == tensor.context['label'].shape[0])
                model.compile('adam', 'mse', metrics=[keras.metrics.MeanAbsoluteError()])
                metrics = model.evaluate(tensor, verbose=0)
                self.assertTrue(isinstance(metrics, list))
                del model

    def test_interpet(self):

        def get_model(tensor):
            inputs = layers.Input(tensor.spec)
            x = layers.NodeEmbedding(32)(inputs)
            x = layers.EdgeEmbedding(32)(x)
            x = layers.GTConv(32)(x)
            x = layers.GTConv(32)(x)
            x = layers.Readout('sum')(x)
            outputs = keras.layers.Dense(1)(x)
            return models.GraphModel(inputs, outputs)
        
        for i, tensor in enumerate(self.tensors):
            with self.subTest(i=i, functional=True):
                model = get_model(tensor)
                tensor = models.interpret(model, tensor)
                self.assertTrue('saliency' in tensor.node)

                tensor = tensor.update({'node': {'saliency': None}})
                tensor = models.saliency(model, tensor)
                self.assertTrue('feature_saliency' in tensor.node)

    def test_embedding(self):

        units = 32

        def get_model(tensor):
            inputs = layers.Input(tensor.spec)
            x = layers.NodeEmbedding(units)(inputs)
            x = layers.EdgeEmbedding(units)(x)
            x = layers.GTConv(units)(x)
            x = layers.GTConv(units)(x)
            x = layers.Readout('sum')(x)
            outputs = keras.layers.Dense(1)(x)
            return models.GraphModel(inputs, outputs)
        
        for i, tensor in enumerate(self.tensors):
            with self.subTest(i=i, functional=True):
                model = get_model(tensor)
                out = model.embedding()(tensor)
                self.assertTrue(out.shape[0] == tensor.context['size'].shape[0])
                self.assertTrue(out.shape[1] == units)