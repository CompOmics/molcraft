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
    data, groups = _prepare_data(
        data, groups, shuffle=shuffle, random_seed=random_seed
    )
    indices = np.unique(groups, return_index=True)[1]
    unique_groups = [groups[i] for i in sorted(indices)]
    size = len(unique_groups) # num examples or num groups

    if not train_size and not test_size:
        raise ValueError(
            f'Found both `train_size` and `test_size` to be `None`, '
            f'specify at least one of them.'
        )
    if isinstance(test_size, float):
        test_size = int(size * test_size)
        if test_size < 1:
            raise ValueError(
                f'`test_size` too small: obtained {test_size} examples.'
            )
    if isinstance(train_size, float):
        train_size = int(size * train_size)
        if train_size < 1:
            raise ValueError(
                f'`train_size` too small: obtained {train_size} examples.'
            )
    if isinstance(validation_size, float):
        validation_size = int(size * validation_size)
        if validation_size < 1:
            raise ValueError(
                f'`validation_size` too small: obtained {validation_size} examples.'
            )
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

    train_mask = np.isin(groups, unique_groups[:train_size])
    test_mask = np.isin(groups, unique_groups[-test_size:])
    if not validation_size:
        return data[train_mask], data[test_mask]
    validation_mask = np.isin(groups, unique_groups[train_size:-test_size])
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
    data, groups = _prepare_data(
        data, groups=groups, shuffle=shuffle, random_seed=random_seed
    )
    indices = np.unique(groups, return_index=True)[1]
    unique_groups = [groups[i] for i in sorted(indices)]
    num_groups = len(unique_groups) # num examples or num groups
    
    if num_splits > num_groups:
        raise ValueError(
            f'`num_splits` ({num_splits}) must not be greater than'
            f'the data size or the number of groups ({num_groups}).'
        )

    unique_groups_splits = np.array_split(unique_groups, num_splits)

    for k in range(num_splits):
        test_groups = unique_groups_splits[k]
        test_mask = np.isin(groups, test_groups)
        train_mask = ~test_mask
        yield data[train_mask], data[test_mask]

def _prepare_data(
    data: pd.DataFrame | np.ndarray,
    groups: str | np.ndarray | None = None,
    shuffle: bool = False,
    random_seed: int | None = None,
) -> pd.DataFrame | np.ndarray:
    
    if not isinstance(data, (pd.DataFrame, np.ndarray)):
        raise ValueError(f'Unsupported `data` type ({type(data)}).')
    
    if shuffle:
        if isinstance(data, pd.DataFrame):
            data = data.sample(
                frac=1., replace=False, random_state=random_seed
            )
        else:
            np.random.seed(random_seed)
            np.random.shuffle(data)

    if isinstance(groups, str):
        groups = data[groups].values
    elif groups is None:
        groups = np.arange(len(data))

    if not isinstance(groups[0], int):
        to_int = {s: i for (i, s) in enumerate(dict.fromkeys(groups))}
        groups = [to_int[group] for group in groups]

    return data, groups