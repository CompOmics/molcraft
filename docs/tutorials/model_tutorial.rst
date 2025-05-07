Part 2: Models
====================

While the featurizer converts a SMILES or InChI encoding of a molecule to a molecular graph, the model 
converts (or 'maps') a molecular graph to some predictions, using a sequence of layers. The mapping from 
graph to prediction is a complex one, including complex graph neural network layers propagating information 
within the graph in a learnable way, using thousands or even millions of so-called weights. These weights 
allow the model to learn meaningful and powerful contextual information resulting in (hopefully) accurate 
predictions. 

There are mainly three submodules to consider for modeling: `records`, `layers` and `models`.

2.1 - Records (**records.py**)
---------------------------------

Records are TensorFlow records (in **.tfrecord** format) that store molecular graphs on disk, which can later 
be loaded as a `tf.data.TFRecordDataset` for efficient input to models. 

**It is recommended to utilize TF records** instead of generating the entire molecular graph to memory (RAM). 
Concretely, instead of generating all molecular (sub)graphs at once as follows:

.. code:: python 

    # data = [(SMILES_1, LABEL_1), ..., (SMILES_N, LABEL_N)]
    graph_tensor = featurizer(data) # May run out of memory if len(data) >> 10000
    dataset = tf.data.Dataset.from_tensor_slices(graph_tensor).batch(32).prefetch(-1)
    model.fit(dataset)

write molecular (sub)graphs to disk and load it as a `tf.data.TFRecordDataset`:

.. code:: python 

    from molcraft import records 

    # data = [(SMILES_1, LABEL_1), ..., (SMILES_N, LABEL_N)]
    records.write(data, featurizer, '/path/to/records/', overwrite=False)
    dataset = records.read('/path/to/records/').batch(32).prefetch(-1)
    model.fit(dataset)


2.2 - Layers (**layers.py**)
---------------------------------

.. code:: python 

    from molcraft import layers 

    graph_tensor = layers.NodeEmbedding(dim=128)(graph_tensor)
    graph_tensor = layers.GraphTransformer(units=128)(graph_tensor)
    graph_tensor = layers.GraphTransformer(units=128)(graph_tensor)
    tensor = layers.Readout(mode='mean')(graph_tensor)


2.3 - Models (**models.py**)
---------------------------------

.. code:: python 

    from molcraft import models 

    model = models.GraphModel.from_layers(
        [
            layers.Input(train_dataset.element_spec),
            layers.NodeEmbedding(dim=128),
            layers.EdgeEmbedding(dim=128),
            layers.GraphTransformer(units=128),
            layers.GraphTransformer(units=128),
            layers.GraphTransformer(units=128),
            layers.GraphTransformer(units=128),
            layers.Readout(mode='mean'),
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(units=1024, activation='relu'),
            keras.layers.Dense(1)
        ]
    )

    model.fit(train_dataset, epochs=10)
    scores = model.evaluate(test_dataset)
    preds = model.predict(test_dataset)


