import warnings
import keras
import numpy as np
import tensorflow as tf

from molcraft import tensors 


def as_dataset(
    x: tensors.GraphTensor,
    /,
    *,
    shuffle: bool = False,
    take: int = None,
    batch_size: int = None,
    random_seed: int = None
) -> tf.data.Dataset:
    if not isinstance(x, tf.data.Dataset):
        ds = tf.data.Dataset.from_tensor_slices(x)
    else:
        ds = x
    if not tensors.is_scalar(ds.element_spec):
        raise ValueError(
            'Input to `dataset` must be a `GraphTensor` instance '
            'or an unbatched `tf.data.Dataset` instance.'
        )
    if shuffle:
        buffer_size = _size(ds) or 1000
        ds = ds.shuffle(buffer_size, seed=random_seed)
    if take:
        ds = ds.repeat().take(take)
    if batch_size:
        ds = ds.batch(batch_size, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds

def blend_datasets(
    datasets: list[tf.data.Dataset] | dict[str, tf.data.Dataset],
    weights: list[float] | dict[str, float] = None,
    *,
    shuffle: bool = False,
    take: int = None,
    batch_size: int = None,
    random_seed: int = None,
) -> tf.data.Dataset:

    weights = weights or None 
    
    if weights and type(weights) != type(datasets):
        raise ValueError(
            f'`datasets` ({type(datasets)}) and `weights` ({type(weights)}) '
            'must be of the same type.'
        )
        
    if isinstance(datasets, dict):
        datasets, weights = _unpack_datasets_and_weights(datasets, weights)
        
    datasets = [
        as_dataset(
            ds, 
            shuffle=shuffle,
            take=take,
            batch_size=None,
            random_seed=random_seed
        ) 
        for ds in datasets
    ]
    ds = tf.data.Dataset.sample_from_datasets(
        datasets=datasets,
        weights=weights,
        seed=random_seed,
        stop_on_empty_dataset=False,
        rerandomize_each_iteration=True,
    )
    return as_dataset(
        ds, 
        shuffle=False, 
        take=take,
        batch_size=batch_size, 
        random_seed=random_seed,
    )

def context_filter(
    field: str, 
    *,
    include: list[int | str] = None, 
    exclude: list[int | str] = None,
) -> callable:
    
    if (include and exclude) or (not include and not exclude):
        raise ValueError(
            'Please specify one of: [`include`, `exclude`].'
        )

    inclusive = include is not None 
    values = include if inclusive else exclude

    if not isinstance(values, (list, tuple, np.ndarray)):
        values = [values]

    values = keras.ops.array(values)
    dtype = values.dtype

    def filter_function(x: tensors.GraphTensor) -> bool:
        context_value = keras.ops.cast(x.context[field], dtype)
        is_in = keras.ops.any(keras.ops.equal(values, context_value))
        return is_in if inclusive else keras.ops.logical_not(is_in)

    return filter_function

def _size(dataset: tf.data.Dataset) -> int | None:
    size = dataset.cardinality()
    if size < 0:
        return None
    return size

def _unpack_datasets_and_weights(
    datasets: list[tf.data.Dataset] | dict[str, tf.data.Dataset],
    weights: list[float] | dict[str, float] = None,
) -> tuple[list[tf.data.Dataset], list[float]]:
    if weights:
        mismatched_weights = weights.keys() - datasets.keys()
        if mismatched_weights:
            raise ValueError(
                f'Found `weights` that do not exist in `datasets`: {mismatched_weights}.'
            )
        missing_weights = datasets.keys() - weights.keys()
        if missing_weights:
            current_sum_weights = sum(weights.values())
            if current_sum_weights >= 1:
                raise ValueError(
                    '`datasets` and `weights` keys must match if the sum '
                    'of the weights is equal to or greater than 1.'
                )
            fill_value = (1 - current_sum_weights) / len(missing_weights)
            for key in missing_weights:
                weights[key] = fill_value
        weights = [weights[key] for key in datasets]
    datasets = [datasets[key] for key in datasets]
    return datasets, weights