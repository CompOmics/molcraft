import warnings
import os
import math
import glob
import time
import typing 
import random
import tensorflow as tf
import numpy as np
import pandas as pd
import multiprocessing as mp

from molcraft import tensors

if typing.TYPE_CHECKING:
    from molcraft import featurizers


def write(
    inputs: list[str | tuple], 
    featurizer: 'featurizers.GraphFeaturizer',
    path: str,
    shuffle: bool = False,
    exist_ok: bool = False,
    overwrite: bool = False, 
    num_files: typing.Optional[int] = None, 
    num_processes: typing.Optional[int] = None,
    multiprocessing: bool = False,
    device: str = '/cpu:0',
    random_seed: int | None = None,
) -> None:
    
    if os.path.isdir(path):
        if not exist_ok:
            raise FileExistsError(f'Records already exist: {path}')
        if not overwrite:
            return
        else:
            _remove_files(path)
    else:
        os.makedirs(path)

    with tf.device(device):
        
        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            inputs = list(inputs.iterrows())

        if isinstance(inputs, tuple):
            inputs = list(inputs)

        if shuffle:
            rng = random.Random(random_seed)
            rng.shuffle(inputs)
            
        example = featurizer._call(inputs[0])
        save_spec(os.path.join(path, 'spec.pb'), example.spec)

        if num_processes is None and multiprocessing:
            num_processes = mp.cpu_count()

        if num_files is None:
            num_examples_per_file = 1_000
            num_files = min(len(inputs), max(1, math.ceil(len(inputs) / num_examples_per_file)))
            
        num_examples = len(inputs)
        chunk_sizes = [0] * num_files
        for i in range(num_examples):
            chunk_sizes[i % num_files] += 1
        
        input_chunks = []
        start_indices = []
        current_index = 0
        for size in chunk_sizes:
            input_chunks.append(inputs[current_index: current_index + size])
            start_indices.append(current_index)
            current_index += size 
        
        assert current_index == num_examples
        
        paths = [
            os.path.join(path, f'tfrecord-{i:06d}.tfrecord')
            for i in range(num_files)
        ]
        
        if not multiprocessing:
            for path, input_chunk, start_index in zip(paths, input_chunks, start_indices):
                _write_tfrecord(input_chunk, path, featurizer, start_index)
            return
        
        processes = []
        
        for path, input_chunk, start_index in zip(paths, input_chunks, start_indices):
        
            while len(processes) >= num_processes:
                for process in processes:
                    if not process.is_alive():
                        processes.remove(process)
                else:
                    time.sleep(0.1)
                    continue
                    
            process = mp.Process(
                target=_write_tfrecord,
                args=(input_chunk, path, featurizer, start_index)
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()         
    
def read(
    path: str, 
    shuffle: bool = False,
    shuffle_files: bool = False,
    shuffle_buffer_size: int = 1000,
    dynamic_file_discovery: bool = False,
    random_seed: int | None = None,
) -> tf.data.Dataset:
    spec = load_spec(os.path.join(path, 'spec.pb'))
    if dynamic_file_discovery:
        def get_filenames():
            filenames = sorted(glob.glob(os.path.join(path, '*.tfrecord')))
            if (shuffle_files or shuffle):
                rng = random.Random(random_seed)
                rng.shuffle(filenames)
            return filenames
        ds = tf.data.Dataset.from_generator(
            get_filenames, output_signature=tf.TensorSpec(shape=(), dtype=tf.string)
        )
    else:
        ds = tf.data.Dataset.list_files(
            os.path.join(path, '*.tfrecord'),
            shuffle=(shuffle_files or shuffle),
            seed=random_seed,
        )
    ds = tf.data.TFRecordDataset(
        ds,
        num_parallel_reads=tf.data.AUTOTUNE
    )
    if dynamic_file_discovery:
        ds = ds.ignore_errors()
    ds = ds.map(
        lambda x: _parse_example(x, spec),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    if not tensors.is_scalar(spec):
        ds = ds.unbatch()
    if shuffle:
        ds = ds.shuffle(shuffle_buffer_size, seed=random_seed)
    return ds

def save_spec(path: str, spec: tensors.GraphTensor.Spec) -> None:
    proto = spec.experimental_as_proto()
    with open(path, 'wb') as fh:
        fh.write(proto.SerializeToString())

def load_spec(path: str) -> tensors.GraphTensor.Spec:
    with open(path, 'rb') as fh:
        serialized_proto = fh.read()
    spec = tensors.GraphTensor.Spec.experimental_from_proto(
        tensors.GraphTensor.Spec
        .experimental_type_proto()
        .FromString(serialized_proto)
    )
    return spec
    
def _write_tfrecord(
    inputs: list, 
    path: str,
    featurizer: 'featurizers.GraphFeaturizer', 
    start_index: int,
) -> None:
    with tf.io.TFRecordWriter(path) as writer:
        for i, x in enumerate(inputs):
            try:
                tensor = featurizer._call(x)
                serialized = _serialize_example(tensor)
                writer.write(serialized)
            except Exception as e:
                index = getattr(x, 'Index', (i + start_index))
                warnings.warn(
                    f'Could not write record for index {index}, '
                    f'proceeding without it. Exception raised:\n{e}'
                )

def _serialize_example(tensor):
    flat_values = tf.nest.flatten(tensor, expand_composites=True)
    flat_values = [
        tf.io.serialize_tensor(value).numpy() for value in flat_values
    ]
    feature = tf.train.Feature(
        bytes_list=tf.train.BytesList(value=flat_values)
    )
    example_proto = tf.train.Example(
        features=tf.train.Features(feature={'feature': feature})
    )
    return example_proto.SerializeToString()
    
def _parse_example(
    x: tf.Tensor, 
    spec: tensors.GraphTensor.Spec
) -> tf.Tensor:
    out = tf.io.parse_single_example(
        x, features={'feature': tf.io.RaggedFeature(tf.string)})['feature']
    out = [
        tf.ensure_shape(tf.io.parse_tensor(x[0], s.dtype), s.shape) 
        for (x, s) in zip(
            tf.split(out, len(tf.nest.flatten(spec, expand_composites=True))), 
            tf.nest.flatten(spec, expand_composites=True)
        )
    ]
    out = tf.nest.pack_sequence_as(spec, tf.nest.flatten(out), expand_composites=True)
    return out

def _remove_files(path):
    for filename in os.listdir(path):
        if filename.endswith('tfrecord') or filename == 'spec.pb':
            filepath = os.path.join(path, filename)
            os.remove(filepath)
