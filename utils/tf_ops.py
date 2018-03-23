import tensorflow as tf

def tf_cosine_distance(a, b):
    epsilon = 1E-6
    num = tf.diag_part(tf.matmul(a, tf.transpose(b)))
    den = tf.maximum(tf.sqrt(tf.reduce_sum(tf.square(a), 1)) * tf.sqrt(tf.reduce_sum(tf.square(b), 1)), epsilon)
    return 1 - ((tf.div(num, den) + 1) / 2)

def compute_triplet_loss(anchor_feature, positive_feature, negative_feature, margin):
    """
    Compute the contrastive loss as in
    Euclidean: L = || f_a - f_p ||^2 - || f_a - f_n ||^2 + m
    Cosine:    L = cos_distance(f_a, f_p) - cos_distance(f_a, f_n) + m
    """
    # d_p = tf.square(tf.subtract(anchor_feature, positive_feature))
    # d_n = tf.square(tf.subtract(anchor_feature, negative_feature))
    # loss = tf.maximum(0., d_p - d_n + margin)

    d_p = tf_cosine_distance(anchor_feature, positive_feature)
    d_n = tf_cosine_distance(anchor_feature, negative_feature)
    loss = tf.maximum(0., d_p - d_n + margin)

    return tf.reduce_mean(loss)
