<img src="https://github.com/akensert/molcraft/blob/main/docs/_static/molcraft-logo.png" alt="molcraft-logo">

**Deep Learning on Molecules**: A Minimalistic GNN package for Molecular ML. 

> [!NOTE]  
> In progress/Unfinished.

## Highlights
- Compatible with **Keras 3**
- Customizable and serializable **featurizers**
- Customizable and serializable **layers** and **models**
- Customizable **GraphTensor**
- Fast and efficient featurization of molecular graphs
- Fast and efficient input pipelines using TF **records**

## Examples 

```python
from molcraft import features
from molcraft import descriptors
from molcraft import featurizers 
from molcraft import layers
from molcraft import models 
import keras

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

graph = featurizer([('N[C@@H](C)C(=O)O', 2.0), ('N[C@@H](CS)C(=O)O', 1.0)])
print(graph)

model = models.GraphModel.from_layers(
    [
        layers.Input(graph.spec),
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

pred = model(graph)
print(pred)

# featurizers.save_featurizer(featurizer, '/tmp/featurizer.json')
# models.save_model(model, '/tmp/model.keras')

# loaded_featurizer = featurizers.load_featurizer('/tmp/featurizer.json')
# loaded_model = models.load_model('/tmp/model.keras')
```

