import warnings
import numpy as np
import pandas as pd
import typing


def split(
    data: pd.DataFrame | np.ndarray,
    *,
    train_size: float | None = None,
    validation_size: float | None = None, 
    test_size: float | None = None,
    groups: str | np.ndarray = None,
    shuffle: bool = False, 
    random_seed: int | None = None,
) -> tuple[np.ndarray | pd.DataFrame, ...]:
    """Splits the dataset into subsets.

    Args:
        data: 
            A pd.DataFrame or np.ndarray object.
        train_size:
            The size of the train set.
        validation_size:
            The size of the validation set.
        test_size:
            The size of the test set.
        groups:
            The groups to perform the splitting on.
        shuffle:
            Whether the dataset should be shuffled prior to splitting.
        random_seed:
            The random state/seed. Only applicable if shuffling.
    """
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError(f'Unsupported `data` type ({type(data)}).')

    if isinstance(groups, str):
        groups = data[groups].values 
    elif groups is None:
        groups = np.arange(len(data))
    
    indices = np.unique(groups)
    size = len(indices)

    if not train_size and not test_size:
        raise ValueError(
            f'Found both `train_size` and `test_size` to be `None`, '
            f'specify at least one of them.'
        )
    if isinstance(test_size, float):
        test_size = int(size * test_size)
    if isinstance(train_size, float):
        train_size = int(size * train_size)
    if isinstance(validation_size, float):
        validation_size = int(size * validation_size)
    elif not validation_size:
        validation_size = 0

    if not train_size:
        train_size = (size - test_size - validation_size)
    if not test_size:
        test_size = (size - train_size - validation_size)
    
    remainder = size - (train_size + validation_size + test_size)
    if remainder < 0:
        raise ValueError(
            f'subset sizes added up to more than the data size.'
        )
    train_size += remainder

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_mask = np.isin(groups, indices[:train_size])
    test_mask = np.isin(groups, indices[-test_size:])
    if not validation_size:
        return data[train_mask], data[test_mask]
    validation_mask = np.isin(groups, indices[train_size:-test_size])
    return data[train_mask], data[validation_mask], data[test_mask]
    
def cv_split(
    data: pd.DataFrame | np.ndarray,
    num_splits: int = 10,
    groups: str | np.ndarray = None,
    shuffle: bool = False, 
    random_seed: int | None = None,
) -> typing.Iterator[
        tuple[np.ndarray | pd.DataFrame, np.ndarray | pd.DataFrame]
    ]:
    """Splits the dataset into cross-validation folds.

    Args:
        data: 
            A pd.DataFrame or np.ndarray object.
        num_splits:
            The number of cross-validation folds.
        groups:
            The groups to perform the splitting on.
        shuffle:
            Whether the dataset should be shuffled prior to splitting.
        random_seed:
            The random state/seed. Only applicable if shuffling.
    """
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError(f'Unsupported `data` type ({type(data)}).')
    
    if isinstance(groups, str):
        groups = data[groups].values
    elif groups is None:
        groups = np.arange(len(data))

    indices = np.unique(groups)
    size = len(indices)
    
    if num_splits > size:
        raise ValueError(
            f'`num_splits` ({num_splits}) must not be greater than'
            f'the data size or the number of groups ({size}).'
        )
    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    indices_splits = np.array_split(indices, num_splits)

    for k in range(num_splits):
        test_indices = indices_splits[k]
        test_mask = np.isin(groups, test_indices)
        train_mask = ~test_mask
        yield data[train_mask], data[test_mask]
