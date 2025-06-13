
.. image:: _static/molcraft-logo.png
   :alt: molcraft-logo
   :align: center
   :width: 100%

|
| **Deep Learning on Molecules**: A Minimalistic GNN package for Molecular ML. 
|

Highlights
-----------------

- Compatible with **Keras 3**
- Simplified API
- Fast featurization
- Modular graph **layers**
- Serializable graph **featurizers** and **models**
- Flexible **GraphTensor**


Installation
-------------------

Install the pre-release of molcraft via pip:

.. code:: bash

  pip install molcraft --pre


with GPU support:

.. code:: bash
  
  pip install molcraft[gpu] --pre


------------------------

.. toctree::
  :glob:
  :maxdepth: 1
  :caption: API

  api/*


.. toctree::
  :glob:
  :maxdepth: 1
  :caption: Tutorials

  tutorials/featurizer_tutorial
  tutorials/model_tutorial
  tutorials/complete_tutorial

