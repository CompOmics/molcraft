import unittest 
import numpy as np

from molcraft import chem


class TestChem(unittest.TestCase):

    def setUp(self):
        self.smiles = [
            "N[C@@H](CC(=O)N)C(=O)O",
            "N1[C@@H](CCC1)C(=O)O",
        ] 

