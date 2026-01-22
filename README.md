<img src="https://github.com/akensert/molcraft/blob/main/docs/_static/molcraft-logo.png" alt="molcraft-logo" width="90%">

**Deep Learning on Molecules**: Graph Neural Networks for Molecular Machine Learning.

## Examples

### Context-Aware Graph Neural Network

Implement a context-aware graph neural network by embedding context features in the super node.
The super node is a virtual node bidirectionally linked to all atomic nodes,
allowing both efficient information propagation and inclusion of context features.
Context features may be continuous or discrete (categorical); for discrete context features, specify
the number of categories expected via `num_categories` of the `AddContext` layer.

```python
from molcraft import features
from molcraft import featurizers 
from molcraft import layers
from molcraft import models

import keras
import pandas as pd

featurizer = featurizers.MolGraphFeaturizer(
    atom_features=[
        features.AtomType(),
        features.NumHydrogens(),
        features.Degree(),
    ],
    bond_features=[
        features.BondType(),
        features.IsRotatable(),
    ],
    super_node=True,
    self_loops=True,
)

df = pd.DataFrame({
    'smiles': [
        'N[C@@H](C)C(=O)O', 'N[C@@H](CS)C(=O)O' 
    ],
    'label': [3.5, -1.5],
    'ph': [7.2, 4.5],
    'temperature': [35., 45.],
})

graph = featurizer(df)

model = models.GraphModel.from_layers(
    [
        layers.Input(graph.spec),
        layers.NodeEmbedding(dim=128),
        layers.EdgeEmbedding(dim=128),
        layers.AddContext(field='ph'),
        layers.AddContext(field='temperature'),
        layers.GraphConv(units=128),
        layers.GraphConv(units=128),
        layers.GraphConv(units=128),
        layers.GraphConv(units=128),
        layers.Readout(mode='mean'),
        keras.layers.Dense(units=1024, activation='elu'),
        keras.layers.Dense(units=1024, activation='elu'),
        keras.layers.Dense(1)
    ]
)

model.compile(
    keras.optimizers.Adam(1e-4), keras.losses.MeanSquaredError()
)
model.fit(graph, epochs=30)
pred = model.predict(graph)

# Uncomment below to save and load model (including featurizer)
# featurizers.save_featurizer(featurizer, '/tmp/featurizer.json')
# models.save_model(model, '/tmp/model.keras')

# loaded_featurizer = featurizers.load_featurizer('/tmp/featurizer.json')
# loaded_model = models.load_model('/tmp/model.keras')
```

### Hybrid Model for Peptides

Implement a GNN-RNN hybrid model for peptides.

```python
from molcraft import features
from molcraft import featurizers 
from molcraft import layers
from molcraft import models

import keras
import pandas as pd

featurizer = featurizers.PeptideGraphFeaturizer(
    atom_features=[
        features.AtomType(),
        features.NumHydrogens(),
        features.Degree(),
    ],
    bond_features=[
        features.BondType(),
        features.IsRotatable(),
    ],
)

# Allow modified amino acids:
# featurizer.monomers.update({
#     "C[Carbamidomethyl]": "N[C@@H](CSCC(=O)N)C(=O)O"
# })

df = pd.DataFrame({
    'sequence': [
        'CYIQNCPLG', 'KTTKS' 
    ],
    'label': [1.0, 0.0],
})

graph = featurizer(df)

model = models.GraphModel.from_layers(
    [
        layers.Input(graph.spec),
        layers.NodeEmbedding(dim=128),
        layers.EdgeEmbedding(dim=128),
        layers.GraphConv(units=128),
        layers.GraphConv(units=128),
        layers.GraphConv(units=128),
        layers.GraphConv(units=128),
        layers.PeptideReadout(),
        keras.layers.Masking(),
        keras.layers.Bidirectional(
            keras.layers.LSTM(units=128, return_sequences=True)
        ),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(units=1024, activation='elu'),
        keras.layers.Dense(units=1024, activation='elu'),
        keras.layers.Dense(1, activation='sigmoid')
    ]
)

model.compile(
    keras.optimizers.Adam(1e-4), keras.losses.BinaryCrossentropy()
)
model.fit(graph, epochs=30)
pred = model.predict(graph)

# Uncomment below to save and load model (including featurizer)
# featurizers.save_featurizer(featurizer, '/tmp/featurizer.json')
# models.save_model(model, '/tmp/model.keras')

# loaded_featurizer = featurizers.load_featurizer('/tmp/featurizer.json')
# loaded_model = models.load_model('/tmp/model.keras')
```

## Installation

For CPU users:

```bash
pip install molcraft
```

For GPU users:
```bash
pip install molcraft[gpu]
```