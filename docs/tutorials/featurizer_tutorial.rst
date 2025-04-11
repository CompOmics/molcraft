Part I: Featurizers
====================

A featurizer simply converts a SMILES or InChI encoding of a molecule to a molecular graph. 

There are three submodules to consider for featurization: `features`, `descriptors` and `featurizers`.

1. Features (**features.py**)
--------------------------------------

A feature computes some specific feature of a molecule, typically an atom or a bond feature.
For instance, `features.AtomType` one-hot encodes atoms based on their type. 
Adding `features.AtomType` to the list of features (see below) results in node features encoding
information about the type of atom. The more features, the more information the graph
neural network model can work with.

2. Descriptors (**descriptors.py**)
--------------------------------------
A descriptor is similar to a feature, but instead of computing an atom or a bond feature,
it computes a molecule feature. For instance, `descriptors.MolTPSA` encodes total polar
surface are of the molecule. Adding descriptors such as `descriptors.MolTPSA` to the 
list of descriptors (see below) results in context features encoding useful information about the 
molecule. This information can later be embedded in the molecular graph via `layers.NodeEmbedding`.

3. Featurizers (**featurizers.py**)
--------------------------------------
As mentioned, featurizers convert SMILES or InChI encodings of a molecule to a molecular graph.
This graph, encoded as a `GraphTensor`, can then be passed to any `layers.GraphLayer` or 
`layers.GraphConv` layer which subsequently propagates information within the molecular graph.

Below is an example of how to construct a `featurizers.MolGraphFeaturizer` to generate a molecular graph,
and then subsequently embed that molecular graph. 

.. code:: python

    from molcraft import featurizers 
    from molcraft import features
    from molcraft import descriptors
    from molcraft import layers

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
        descriptors=[
            molcraft.descriptors.MolWeight(),
            molcraft.descriptors.MolTPSA(),
            molcraft.descriptors.MolLogP(),
            molcraft.descriptors.NumHeavyAtoms(),
            molcraft.descriptors.NumHydrogenDonors(),
            molcraft.descriptors.NumHydrogenAcceptors(),
            molcraft.descriptors.NumRotatableBonds(),
            molcraft.descriptors.NumRings(),
        ],
        super_atom=True,
        self_loops=False,
    )

    molgraph = featurizer(['N[C@@H](C)C(=O)O', 'N[C@@H](CS)C(=O)O'])
    print(molgraph)

    molgraph_updated = layers.NodeEmbedding(dim=128)(molgraph)
    print(molgraph_updated)

To embed context in the molecular graph it is necessary to specify `super_atom=True` as it adds 
additional nodes to the graph, which can later (via `layers.NodeEmbedding`) be filled with
context features. This "super atom" (or super/virtual node) is a special node in the molecular graph 
which does not correspond to an atom, but to something else. For instance, this special node may 
encode additional information about the molecule (such as descriptors, as in the example above) or 
additional information about the environment (e.g., for QSRR, the chromatographic instrument and parameters).

Furthermore, for 3D molecular graphs, a `featurizers.MolGraphFeaturizer3D` is also implemented:

.. code:: python

    from molcraft import featurizers 
    from molcraft import features
    from molcraft import descriptors
    from molcraft import conformers
    from molcraft import layers

    featurizer = featurizers.MolGraphFeaturizer3D(
        atom_features=[
            features.AtomType(),
            features.TotalNumHs(),
            features.Degree(),
        ],
        descriptors=[
            molcraft.descriptors.MolWeight(),
            molcraft.descriptors.MolTPSA(),
            molcraft.descriptors.MolLogP(),
            molcraft.descriptors.NumHeavyAtoms(),
            molcraft.descriptors.NumHydrogenDonors(),
            molcraft.descriptors.NumHydrogenAcceptors(),
            molcraft.descriptors.NumRotatableBonds(),
            molcraft.descriptors.NumRings(),
        ],
        conformer_generator=conformers.ConformerEmbedder(
            method='ETKDGv3'
            num_conformers=5
        ),
        super_atom=True,
        self_loops=False,
        radius=6.0,
    )

    molgraph = featurizer(['N[C@@H](C)C(=O)O', 'N[C@@H](CS)C(=O)O'])
    print(molgraph)

    molgraph_updated = layers.NodeEmbedding(dim=128)(molgraph)
    print(molgraph_updated)

There are mainly two differences between a typical (non-3D) molecular graph and a 3D molecular graph: 
(1) the molecular graph encodes cartesian coordinates; and (2) the edges of the graph are not
limited by bonds and does not typically encode bond features. Regarding the latter, edges are typically
added if a neighboring atom is within a certain radius in 3D space; and the associated edge features are
by default a one-hot encoding of the number of hops between the two atom-pairs. Notably, the radius 
is in unit 'angstrom', and not the number of bonds in the shortest path between atom pairs (which is the 
case for `featurizers.MolGraphFeaturizer`).

Finally, to include labels (and optionally sample weights) you can simply pass a 2- or 3-tuple to the
featurizer:

.. code:: python 
    
    # Use default parameters
    featurizer = featurizers.MolGraphFeaturizer()

    data = [('N[C@@H](C)C(=O)O', 12.3, 0.5), ('N[C@@H](CS)C(=O)O', 15.6, 0.75)]
    molgraph = featurizer(data)
    print(molgraph)

The molecular graph can now be used to train a graph neural network model (see next).