import math
import keras
import numpy as np
import pandas as pd
import tensorflow as tf
import typing
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
)

from molcraft import featurizers


class PyDataset(keras.utils.PyDataset):

    def __init__(
        self, 
        data: pd.DataFrame, 
        featurizer: featurizers.GraphFeaturizer, 
        batch_size: int = 32, 
        shuffle: bool = False, 
        random_seed: int = None, 
        **kwargs
    ) -> None:
        super().__init__(**kwargs)
        self.data = data
        self.featurizer = featurizer
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.random_seed = random_seed
        self.indices = np.arange(len(self.data))
        if self.shuffle:
            self.on_epoch_end()
        self._element_spec = None

    def on_epoch_end(self) -> None:
        if self.shuffle:
            rng = np.random.default_rng(self.random_seed)
            rng.shuffle(self.indices)
            if self.random_seed is not None:
                self.random_seed += 1

    def __len__(self) -> int:
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, index: int) -> dict[str, dict[str, np.ndarray]]:
        low = index * self.batch_size
        high = min(low + self.batch_size, len(self.data))
        batch_indices = self.indices[low:high]
        batch = self.data.iloc[batch_indices]
        return self.featurizer(batch, _return_graph_tensor=False)

    @property 
    def element_spec(self) -> dict[str, dict[str, tf.TensorSpec]]:
        if not self._element_spec:
            self._element_spec = keras.tree.map_structure(
                lambda x: tf.TensorSpec(shape=(None,) + x.shape[1:], dtype=x.dtype),
                self.__getitem__(0)
            )
        return self._element_spec


def cv_split(
    data: pd.DataFrame,
    num_splits: int = 5,
    *,
    shuffle: bool = False, 
    random_seed: int | None = None,
    group_by: str | None = None,
    stratify_by: str | None = None,
    **kwargs,
) -> typing.Iterator[tuple[pd.DataFrame, pd.DataFrame]]:
    '''Splits a DataFrame into cross-validation folds.

    Args:
        data: 
            The `pd.DataFrame` to be split.
        num_splits: 
            The number of folds to create. Defaults to 5.
        shuffle: 
            Whether to shuffle the data before splitting. 
        random_seed: 
            Seed for the random number generator for reproducibility.
        group_by: 
            Column name used to ensure the same group does not appear 
            in both train and test sets.
        stratify_by: 
            Column name used to ensure folds maintain roughly the same 
            class distribution as the original dataset.
    '''
    if not isinstance(data, pd.DataFrame):
        raise ValueError('`splits` only supports `pd.DataFrame` data.')
    
    if group_by is not None and not isinstance(group_by, str):
        raise ValueError('`group_by` needs to be `str` or `None`.')
    
    if stratify_by is not None and not isinstance(stratify_by, str):
        raise ValueError('`stratify_by` needs to be `str` or `None`.')

    if shuffle:
        data = data.sample(frac=1, random_state=random_seed)

    if group_by is not None and stratify_by is not None:
        kfold = StratifiedGroupKFold(num_splits)
    elif group_by is not None:
        kfold = GroupKFold(num_splits)
    elif stratify_by is not None:
        kfold = StratifiedKFold(num_splits)
    else:
        kfold = KFold(num_splits)

    X = range(len(data))
    y = pd.factorize(data[stratify_by])[0] if stratify_by else None
    groups = pd.factorize(data[group_by])[0] if group_by else None
    
    for train_indices, test_indices in kfold.split(X=X, y=y, groups=groups):
        yield data.iloc[train_indices], data.iloc[test_indices]
