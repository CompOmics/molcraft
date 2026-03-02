import warnings
import numpy as np
import pandas as pd
import typing
from sklearn.model_selection import (
    KFold, StratifiedKFold, GroupKFold, StratifiedGroupKFold
)


def split(
    data: pd.DataFrame,
    test_size: float = 0.2,
    val_size: float | None = None,
    *,
    shuffle: bool = True,
    random_seed: int | None = None,
    group_by: str | None = None,
    stratify_by: str | None = None,
    **kwargs,
) -> tuple[pd.DataFrame, ...]:
    '''Splits a DataFrame into two (Train/Test) or three (Train/Val/Test) sets.

    Args:
        data: 
            The `pd.DataFrame` to be split.
        test_size: 
            The fraction of data to be used for the test set. Default to 0.2.
            Note: the size of the resulting subset may differ significantly
            from the expected size specified by `test_size`.
        val_size: 
            The fraction of data to be used for the validation set. Default to None.
            Note: the size of the resulting subset may differ significantly
            from the expected size specified by `val_size`.
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
    groups = kwargs.pop('groups', None)
    if groups is not None:
        warnings.warn(
            message=(
                '`groups` argument has been deprecated, please use `group_by` instead.'
            ),
            category=DeprecationWarning,
            stacklevel=2
        )
        group_by = groups 

    _num_splits_test = int(round(1 / test_size))

    data_train_val, data_test = next(
        cv_split(
            data, 
            num_splits=_num_splits_test, 
            shuffle=shuffle, 
            random_seed=random_seed,
            group_by=group_by,
            stratify_by=stratify_by
        )
    )
    if not val_size:
        return data_train_val, data_test 

    _adj_val_ratio = val_size / (1 - test_size)
    _num_splits_val = int(round(1 / _adj_val_ratio))

    data_train, data_val = next(
        cv_split(
            data_train_val, 
            num_splits=_num_splits_val, 
            shuffle=False, # already shuffled 
            random_seed=random_seed,
            group_by=group_by,
            stratify_by=stratify_by
        )
    )
    return data_train, data_val, data_test 

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
    groups = kwargs.pop('groups', None)
    if groups is not None:
        warnings.warn(
            message=(
                '`groups` argument has been deprecated, '
                'please use `group_by` instead.'
            ),
            category=DeprecationWarning,
            stacklevel=2
        )
        group_by = groups 

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
