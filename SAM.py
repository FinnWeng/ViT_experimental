import tensorflow as tf


def dual_vector(grads):
    """Returns the solution of max_x y^T x s.t. ||x||_2 <= 1.
    Args:
        y: A pytree of numpy ndarray, vector y in the equation above.
    """

    gradient_norm = tf.math.sqrt(sum(tf.nest.map_structure(lambda x: tf.reduce_sum(tf.math.square(x)), grads)))
    normalized_gradient = tf.nest.map_structure(lambda x: x / gradient_norm, grads)

    return normalized_gradient