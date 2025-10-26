import unittest 
import numpy as np

from molcraft import chem

from .base_test import MolCraftTest


class TestChem(MolCraftTest):

    def setUp(self):
        self.smiles = [
            "N[C@@H](CC(=O)N)C(=O)O",
            "N1[C@@H](CCC1)C(=O)O",
        ] 


if __name__ == '__main__':
    unittest.main()