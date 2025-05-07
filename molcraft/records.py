import os
import math
import glob
import time
import typing 
import tensorflow as tf
import numpy as np
import pandas as pd
import multiprocessing as mp

from molcraft import tensors
from molcraft import featurizers


def write(
    inputs: list[str | tuple], 
    featurizer: featurizers.Featurizer,
    path: str, 
    overwrite: bool = True, 
    num_files: typing.Optional[int] = None, 
    num_processes: typing.Optional[int] = None,
    multiprocessing: bool = False,
    device: str = '/cpu:0'
) -> None:
    
    if os.path.isdir(path) and not overwrite:
        return 
    
    os.makedirs(path, exist_ok=True)

    with tf.device(device):

        if isinstance(inputs, (pd.DataFrame, pd.Series)):
            inputs = inputs.values
        
        if not isinstance(inputs, list):
            inputs = list(inputs)

        example = inputs[0]
        if isinstance(example, (list, np.ndarray)):
            example = tuple(example)
        example = featurizer(example)
        if not isinstance(example, tensors.GraphTensor):
            example = example[0]

        save_spec(os.path.join(path, 'spec.pb'), example.spec)

        if num_processes is None:
            num_processes = mp.cpu_count()

        if num_files is None:
            num_files = min(len(inputs), max(1, math.ceil(len(inputs) / 1_000)))
            
        chunk_size = math.ceil(len(inputs) / num_files)
        num_files = math.ceil(len(inputs) / chunk_size)
        
        paths = [
            os.path.join(path, f'tfrecord-{i:04d}.tfrecord')
            for i in range(num_files)
        ]
        
        input_chunks = [
            inputs[i * chunk_size: (i + 1) * chunk_size]
            for i in range(num_files)
        ]
        
        if not multiprocessing:
            for path, input_chunk in zip(paths, input_chunks):
                _write_tfrecord(input_chunk, path, featurizer)
            return
        
        processes = []
        
        for path, input_chunk in zip(paths, input_chunks):
        
            while len(processes) >= num_processes:
                for process in processes:
                    if not process.is_alive():
                        processes.remove(process)
                else:
                    time.sleep(0.1)
                    continue
                    
            process = mp.Process(
                target=_write_tfrecord,
                args=(input_chunk, path, featurizer)
            )
            processes.append(process)
            process.start()

        for process in processes:
            process.join()         
    
def read(
    path: str, 
    shuffle_files: bool = False
) -> tf.data.Dataset:
    spec = load_spec(os.path.join(path, 'spec.pb'))
    filenames = sorted(glob.glob(os.path.join(path, '*.tfrecord')))
    num_files = len(filenames)
    ds = tf.data.Dataset.from_tensor_slices(filenames)
    if shuffle_files:
        ds = ds.shuffle(num_files)
    ds = ds.interleave(
        tf.data.TFRecordDataset, num_parallel_calls=1)
    ds = ds.map(
        lambda x: _parse_example(x, spec),
        num_parallel_calls=tf.data.AUTOTUNE)
    if not tensors.is_scalar(spec):
        ds = ds.unbatch()
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
    inputs, 
    path: str,
    featurizer: featurizers.Featurizer, 
) -> None:
    
    def _write_example(tensor):
        flat_values = tf.nest.flatten(tensor, expand_composites=True)
        flat_values = [tf.io.serialize_tensor(value).numpy() for value in flat_values]
        feature = tf.train.Feature(bytes_list=tf.train.BytesList(value=flat_values))
        serialized_feature = _serialize_example({'feature': feature})
        writer.write(serialized_feature)

    with tf.io.TFRecordWriter(path) as writer:
        for x in inputs:
            if isinstance(x, (list, np.ndarray)):
                x = tuple(x)
            tensor = featurizer(x)
            if tensor is not None:
                if isinstance(tensor, tensors.GraphTensor):
                    _write_example(tensor)
                else:
                    for t in tensor:
                        _write_example(t)

def _serialize_example(
    feature: dict[str, tf.train.Feature]
) -> bytes:
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()

def _parse_example(
    x: tf.Tensor, 
    spec: tensors.GraphTensor.Spec
) -> tf.Tensor:
    out = tf.io.parse_single_example(
        x, features={'feature': tf.io.RaggedFeature(tf.string)})['feature']
    out = [tf.ensure_shape(tf.io.parse_tensor(x[0], s.dtype), s.shape) for (x, s) in zip(
        tf.split(out, len(tf.nest.flatten(spec, expand_composites=True))), 
        tf.nest.flatten(spec, expand_composites=True))]
    out = tf.nest.pack_sequence_as(spec, tf.nest.flatten(out), expand_composites=True)
    return out