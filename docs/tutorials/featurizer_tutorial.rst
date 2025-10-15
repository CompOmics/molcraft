Part 1: Featurizers
====================

A featurizer simply converts a SMILES or InChI encoding of a molecule to a molecular graph. 

There are three submodules to consider for featurization: `features`, `descriptors` and `featurizers`.

1.1 - Features (**features.py**)
--------------------------------------

A feature encodes a certain feature of a molecule, typically an atom or a bond feature.
For instance, `features.AtomType` one-hot encodes atoms based on their type. 
Adding `features.AtomType` to the list of features (see below) results in node features encoding
information about the type of atom. The more features, the more information the graph
neural network model can work with.

1.2 - Descriptors (**descriptors.py**)
--------------------------------------
A descriptor is similar to a feature, but instead of computing an atom or a bond feature,
it computes a molecule feature. For instance, `descriptors.TotalPolarSurfaceArea` encodes total polar
surface are of the molecule. Adding descriptors such as `descriptors.TotalPolarSurfaceArea` to the 
list of descriptors (see below) results in context features encoding useful information about the 
molecule. This information can later be embedded in the super node of the molecular graph via `layers.NodeEmbedding`.

1.3 - Featurizers (**featurizers.py**)
--------------------------------------
As mentioned, featurizers convert SMILES or InChI encodings of a molecule to a molecular graph.
This graph, encoded as a `GraphTensor`, can then be passed to any `layers.GraphLayer` or 
`layers.GraphConv` layer which subsequently propagates information within the molecular graph.

Below is an example of how to construct a `featurizers.MolGraphFeaturizer` to generate a molecular graph,
and then subsequently embed that molecular graph via `layers.NodeEmbedding`. 

.. code:: python
    
    from molcraft import features
    from molcraft import descriptors
    from molcraft import featurizers 
    from molcraft import layers

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
        molecule_features=[
            descriptors.MolWeight(),
            descriptors.TotalPolarSurfaceArea(),
            descriptors.LogP(),
            descriptors.MolarRefractivity(),
            descriptors.NumHeavyAtoms(),
            descriptors.NumHeteroatoms(),
            descriptors.NumHydrogenDonors(),
            descriptors.NumHydrogenAcceptors(),
            descriptors.NumRotatableBonds(),
            descriptors.NumRings(),
        ],
        super_node=True,
        self_loops=False,
    )

    molgraph = featurizer(['N[C@@H](C)C(=O)O', 'N[C@@H](CS)C(=O)O'])
    print(molgraph)

    molgraph_updated = layers.NodeEmbedding(dim=128)(molgraph)
    print(molgraph_updated)

To embed context in the molecular graph it is necessary to specify `super_node=True` as it adds 
an additional node to the graph, which can then later (via `layers.NodeEmbedding`) be filled with
the context feature.

Furthermore, for 3D molecular graphs, a `featurizers.MolGraphFeaturizer3D` is also implemented:

.. code:: python

    featurizer = featurizers.MolGraphFeaturizer3D(
        atom_features=[
            features.AtomType(),
            features.NumHydrogens(),
            features.Degree(),
        ],
        pair_features=[
            features.PairDistance(),
        ],
        molecule_features=[
            descriptors.MolWeight(),
            descriptors.TotalPolarSurfaceArea(),
            descriptors.LogP(),
            descriptors.MolarRefractivity(),
            descriptors.NumHeavyAtoms(),
            descriptors.NumHeteroatoms(),
            descriptors.NumHydrogenDonors(),
            descriptors.NumHydrogenAcceptors(),
            descriptors.NumRotatableBonds(),
            descriptors.NumRings(),
        ],
        super_node=True,
        self_loops=False,
        radius=6.0,
    )

    molgraph = featurizer(['N[C@@H](C)C(=O)O', 'N[C@@H](CS)C(=O)O'])
    print(molgraph)

    molgraph_updated = layers.NodeEmbedding(dim=128)(molgraph)
    print(molgraph_updated)

The 3D molecular graph adds cartesian coordinates and replaces edges based on bonds with edges 
based on distances. So the nodes of the 3D molecular graph are linked if they are within a certain
radius of each other.

Finally, to include labels (and optionally sample weights) you can simply pass a 2- or 3-tuple to the
featurizer:

.. code:: python 
    
    # Use default arguments
    featurizer = featurizers.MolGraphFeaturizer()
    # Dummy data
    data = [('N[C@@H](C)C(=O)O', 12.3, 0.5), ('N[C@@H](CS)C(=O)O', 15.6, 0.75)] 
    molgraph = featurizer(data)
    print(molgraph)

The molecular graph can now be used to train a graph neural network model (see next).