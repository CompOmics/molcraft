import numpy as np
import pandas as pd


def split(
    data: pd.DataFrame | np.ndarray,
    train_size: float | None = None,
    validation_size: float | None = None, 
    test_size: float = 0.1,
    shuffle: bool = False, 
    random_state: int | None = None,
) -> pd.DataFrame | np.ndarray:
    """Splits dataset into subsets.

    Args:
        data: 
            A pd.DataFrame or np.ndarray object.
        train_size:
            Optional train size, as a fraction (`float`) or size (`int`).
        validation_size:
            Optional validation size, as a fraction (`float`) or size (`int`).
        test_size:
            Required test size, as a fraction (`float`) or size (`int`).
        shuffle:
            Whether the dataset should be shuffled prior to splitting.
        random_state:
            The random state (or seed). Only applicable if shuffling.
    """

    if not isinstance(data, (pd.DataFrame, np.ndarray, list)):
        raise ValueError(
            '`data` needs to be a pd.DataFrame, np.ndarray or a list. '
            f'Found {type(data)}.'
        )
    
    size = len(data)
    
    if test_size is None:
        raise ValueError('`test_size` is required.')
    elif test_size <= 0:
        raise ValueError(
            f'Test size needs to be positive. Found: {test_size}. '
            'Either specify a positive `float` (fraction) or '
            'a positive `int` (size).'
        )
    if train_size is not None and train_size <= 0:
        raise ValueError(
            f'Train size needs to be None or positive. Found: {train_size}. '
            'Either specify `None`, a positive `float` (fraction) or '
            'a positive `int` (size).'
        )
    if validation_size is not None and validation_size <= 0:
        raise ValueError(
            f'Validation size needs to be None or positive. Found: {validation_size}. '
            'Either specify `None`, a positive `float` (fraction) or '
            'a positive `int` (size).'
        )
    
    if isinstance(test_size, float):
        test_size = int(size * test_size)
    if validation_size and isinstance(validation_size, float):
        validation_size = int(size * validation_size)
    elif not validation_size:
        validation_size = 0

    if train_size and isinstance(train_size, float):
        train_size = int(size * train_size)
    elif not train_size:
        train_size = 0

    if not train_size:
        train_size = size - test_size
        if not validation_size:
            train_size -= validation_size

    remainder = size - (train_size + validation_size + test_size)

    if remainder < 0:
        raise ValueError(
            'Sizes of data subsets add up to more than the size of the original data set: '
            f'{size} < ({train_size} + {validation_size} + {test_size})'
        )
    if test_size <= 0:
        raise ValueError(
            f'Test size needs to be greater than 0. Found: {test_size}.'
        )
    if train_size <= 0:
        raise ValueError(
            f'Train size needs to be greater than 0. Found: {train_size}.'
        )
    
    train_size += remainder

    if isinstance(data, pd.DataFrame):
        if shuffle:
            data = data.sample(
                frac=1.0, replace=False, random_state=random_state
            )
        train_data = data.iloc[:train_size]
        test_data = data.iloc[-test_size:]
        if not validation_size:
            return train_data, test_data 
        validation_data = data.iloc[train_size:-test_size]
        return train_data, validation_data, test_data
    
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    np.random.seed(random_state)

    random_indices = np.arange(size)
    np.random.shuffle(random_indices)
    data = data[random_indices]

    train_data = data[:train_size]
    test_data = data[-test_size:]
    if not validation_size:
        return train_data, test_data 
    validation_data = data[train_size:-test_size]
    return train_data, validation_data, test_data
    

    
