import unittest
import pandas as pd
import numpy as np

from molcraft.datasets import split, cv_split


class TestDatasets(unittest.TestCase):

    def setUp(self):
        self.smiles = [f"C1=CC=CC=C1{i}" for i in range(10)] * 10
        self.labels = [0] * 50 + [1] * 50
        self.df = pd.DataFrame({
            'smiles': self.smiles,
            'label': self.labels,
            'extra_data': np.random.rand(100)
        })

    def test_cv_split_fold_count(self):
        num_splits = 5
        folds = list(cv_split(self.df, num_splits=num_splits))
        self.assertEqual(len(folds), num_splits)

    def test_cv_split_group_leakage(self):
        for train, test in cv_split(self.df, group_by='smiles', num_splits=3):
            train_smiles = set(train['smiles'])
            test_smiles = set(test['smiles'])
            overlap = train_smiles.intersection(test_smiles)
            self.assertEqual(len(overlap), 0, f"Leakage detected: {overlap}")

    def test_split_2way_shapes(self):
        train, test = split(self.df, test_size=0.2)
        self.assertEqual(len(train), 80)
        self.assertEqual(len(test), 20)

    def test_stratification_balance(self):
        _, test = split(self.df, test_size=0.2, stratify_by='label', shuffle=True)
        label_counts = test['label'].value_counts()
        self.assertTrue(8 <= label_counts[0] <= 12) 

    def test_reproducibility(self):
        kwargs = {'test_size': 0.2, 'random_seed': 42, 'shuffle': True}
        train1, test1 = split(self.df, **kwargs)
        train2, test2 = split(self.df, **kwargs)
        pd.testing.assert_frame_equal(train1, train2)
        pd.testing.assert_frame_equal(test1, test2)

    def test_stratified_group_split(self):
        train, test = split(
            self.df, 
            test_size=0.2, 
            group_by='smiles', 
            stratify_by='label', 
            shuffle=True, 
            random_seed=42
        )
        train_smiles = set(train['smiles'])
        test_smiles = set(test['smiles'])
        self.assertEqual(len(train_smiles.intersection(test_smiles)), 0)
        label_counts = test['label'].value_counts()
        self.assertTrue(label_counts[0] >= 8 and label_counts[1] >= 8)
        self.assertEqual(len(train) + len(test), len(self.df))
