import unittest
import pandas as pd
import numpy as np

from molcraft import utils


class TestUtils(unittest.TestCase):

    def setUp(self):
        smiles = [f"C1=CC=CC=C1{i}" for i in range(10)] * 10
        labels = [0] * 50 + [1] * 50
        groups = ['A', 'B', 'C', 'D'] * 25
        self.df = pd.DataFrame({
            'smiles': smiles, 'label': labels, 'groups': groups
        })

    def test_cv_split_num_splits(self):
        splits = list(utils.cv_split(self.df, num_splits=5))
        self.assertEqual(len(splits), 5)

    def test_cv_split_group_by(self):
        num_splits = 4
        for train, test in utils.cv_split(self.df, num_splits=num_splits, group_by='groups'):
            train_groups = set(train['groups'].unique())
            test_groups = set(test['groups'].unique())
            self.assertEqual(len(train_groups.intersection(test_groups)), 0)

    def test_cv_split_stratify_by(self):
        num_splits = 5
        for train, test in utils.cv_split(self.df, num_splits=num_splits, stratify_by='label'):
            train_prop = train['label'].mean()
            test_prop = test['label'].mean()
            self.assertAlmostEqual(train_prop, 0.5, delta=0.1)
            self.assertAlmostEqual(test_prop, 0.5, delta=0.1)

    def test_cv_split_group_by_stratify_by(self):
        for train, test in utils.cv_split(self.df, num_splits=2, group_by='groups', stratify_by='label'):
            self.assertEqual(len(set(train['groups']).intersection(set(test['groups']))), 0)
            self.assertAlmostEqual(train['label'].mean(), 0.5, delta=0.1)

    def test_cv_split_shuffle(self):
        seed = 42
        split1 = list(utils.cv_split(self.df, num_splits=2, shuffle=True, random_seed=seed))
        split2 = list(utils.cv_split(self.df, num_splits=2, shuffle=True, random_seed=seed))
        pd.testing.assert_frame_equal(split1[0][0], split2[0][0])