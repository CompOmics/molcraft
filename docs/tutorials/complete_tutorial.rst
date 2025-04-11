Part 3: Complete modeling pipeline
===================================


.. code:: python

    from molcraft import features
    from molcraft import descriptors
    from molcraft import featurizers
    from molcraft import records 
    from molcraft import layers
    from molcraft import models
    from molcraft import datasets
    import keras 
    import pandas as pd 

    df = pd.read_csv('/path/to/dataset.csv')

    data = df[['smiles_col', 'label_col']].values
    # TODO: Allow label masks for missing labels

    train_data, validation_data, test_data = datasets.split(
        data, train_frac=0.8, validation_frac=0.1, test_frac=0.1, shuffle=True, random_state=42,
    )

    featurizer = featurizers.MolGraphFeaturizer(
        atom_features=[
            features.AtomType(),
            features.TotalNumHs(),
            features.Degree(),
        ],
        bond_features=[
            features.BondType(),
            features.IsRotatable(),
        ],
        super_atom=True,
        self_loops=False,
    )

    records.write(train_data, featurizer, '/path/to/records/train/', overwrite=False)
    records.write(validation_data, featurizer, '/path/to/records/validation/', overwrite=False)
    records.write(test_data, featurizer, '/path/to/records/test/', overwrite=False)

    train_dataset = records.load('/path/to/records/train/', shuffle_files=True)
    train_dataset = train_dataset.shuffle(1024).batch(32).prefetch(-1)

    validation_dataset = records.load('/path/to/records/validation/')
    validation_dataset = validation_dataset.batch(128).prefetch(-1)

    test_dataset = records.load('/path/to/records/test/')
    test_dataset = test_dataset.batch(128).prefetch(-1)
    
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
    model.compile(
        keras.optimizers.Adam(1e-3), 
        keras.losses.MeanAbsoluteError(), 
        metrics=[
            keras.metrics.MeanAbsolutePercentageError(name='mape'),
            keras.metrics.MeanSquaredError(name='mse'),
        ]
    )
    model.fit(
        train_dataset, 
        validation_data=validation_dataset,
        epochs=300, # maximum 300 epochs
        callbacks=[
            keras.callbacks.ReduceLROnPlateau(
                factor=0.1,
                patience=10,
            ),
            keras.callbacks.EarlyStopping(
                patience=15,
                restore_best_weights=True
            ),
        ]
    )
    scores = model.evaluate(test_dataset)
    preds = model.predict(test_dataset)
