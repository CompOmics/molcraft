import keras
import numpy as np
import tensorflow as tf
from keras import backend


def gather(
    node_feature: tf.Tensor, 
    edge: tf.Tensor
) -> tf.Tensor:
    if backend.backend() == 'tensorflow':
        return tf.gather(node_feature, edge)
    expected_rank = len(keras.ops.shape(node_feature))
    current_rank = len(keras.ops.shape(edge))
    for _ in range(expected_rank - current_rank):
        edge = keras.ops.expand_dims(edge, axis=-1)
    return keras.ops.take_along_axis(node_feature, edge, axis=0)

def aggregate(
    node_feature: tf.Tensor, 
    edge: tf.Tensor, 
    num_nodes: tf.Tensor,
    mode: str = 'sum',
) -> tf.Tensor:
    if mode == 'mean':
        return segment_mean(
            node_feature, edge, num_nodes, sorted=False
        )
    return keras.ops.segment_sum(
        node_feature, edge, num_nodes, sorted=False
    )

def propagate(
    node_feature: tf.Tensor,
    edge_source: tf.Tensor,
    edge_target: tf.Tensor,
    edge_feature: tf.Tensor | None = None,
    edge_weight: tf.Tensor | None = None,
) -> tf.Tensor:
    num_nodes = keras.ops.shape(node_feature)[0]

    node_feature_source = gather(node_feature, edge_source)
    
    if edge_weight is not None:
        node_feature_source *= edge_weight

    if edge_feature is not None:
        node_feature_source += edge_feature
        
    return aggregate(node_feature, edge_target, num_nodes)

def scatter_update(
    inputs: tf.Tensor,
    indices: tf.Tensor,
    updates: tf.Tensor,
) -> tf.Tensor:
    if indices.dtype == tf.bool:
        indices = keras.ops.stack(keras.ops.where(indices), axis=-1)
    expected_rank = len(keras.ops.shape(inputs))
    current_rank = len(keras.ops.shape(indices))
    for _ in range(expected_rank - current_rank):
        indices = keras.ops.expand_dims(indices, axis=-1)
    return keras.ops.scatter_update(inputs, indices, updates)

def edge_softmax(
    score: tf.Tensor, 
    edge_target: tf.Tensor
) -> tf.Tensor:
    num_segments = keras.ops.cond(
        keras.ops.greater(keras.ops.shape(edge_target)[0], 0),
        lambda: keras.ops.maximum(keras.ops.max(edge_target) + 1, 1),
        lambda: 0
    )
    score_max = keras.ops.segment_max(
        score, edge_target, num_segments, sorted=False
    )
    score_max = gather(score_max, edge_target)
    numerator = keras.ops.exp(score - score_max)
    denominator = keras.ops.segment_sum(
        numerator, edge_target, num_segments, sorted=False
    )
    denominator = gather(denominator, edge_target)
    return numerator / denominator

def edge_weight(
    edge: tf.Tensor,
    edge_weight: tf.Tensor,
) -> tf.Tensor:
    expected_rank = len(keras.ops.shape(edge))
    current_rank = len(keras.ops.shape(edge_weight))
    for _ in range(expected_rank - current_rank):
        edge_weight = keras.ops.expand_dims(edge_weight, axis=-1)
    return edge * edge_weight

def segment_mean(
    data: tf.Tensor,
    segment_ids: tf.Tensor,
    num_segments: int | None = None,
    sorted: bool = False,
) -> tf.Tensor:
    if num_segments is None:
        num_segments = keras.ops.cond(
            keras.ops.greater(keras.ops.shape(segment_ids)[0], 0),
            lambda: keras.ops.max(segment_ids) + 1,
            lambda: 0
        )
    if backend.backend() == 'tensorflow':
        return tf.math.unsorted_segment_mean(
            data=data,
            segment_ids=segment_ids,
            num_segments=num_segments
        )
    x = keras.ops.segment_sum(
        data=data, 
        segment_ids=segment_ids, 
        num_segments=num_segments,
        sorted=sorted
    )
    sizes = keras.ops.cast(
        keras.ops.bincount(segment_ids, minlength=num_segments), 
        dtype=x.dtype
    )
    return x / sizes[:, None]

def gaussian(
    x: tf.Tensor, 
    mean: tf.Tensor, 
    std: tf.Tensor
) -> tf.Tensor:
    expected_rank = len(keras.ops.shape(x))
    current_rank = len(keras.ops.shape(mean))
    for _ in range(expected_rank - current_rank):
        mean = keras.ops.expand_dims(mean, axis=0)
        std = keras.ops.expand_dims(std, axis=0)
    a = (2 * np.pi) ** 0.5
    return keras.ops.exp(-0.5 * (((x - mean) / std) ** 2)) / (a * std)

def euclidean_distance(
    x1: tf.Tensor, 
    x2: tf.Tensor, 
    axis: int = -1
) -> tf.Tensor:
    relative_distance = keras.ops.subtract(x1, x2)
    return keras.ops.sqrt(
        keras.ops.sum(
            keras.ops.square(relative_distance), 
            axis=axis, 
            keepdims=True
        )
    )