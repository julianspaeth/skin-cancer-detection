import tensorflow as tf


def l1_loss(logits, labels):
    subs = tf.subtract(labels, logits)
    error_per_image = tf.reduce_sum(tf.abs(subs), axis=1)
    error_per_image_reg = tf.multiply(error_per_image, _regularize(labels))
    return tf.reduce_sum(error_per_image_reg)


def l2_loss(logits, labels):
    subs = tf.subtract(labels, logits)
    error_per_image = tf.reduce_sum(tf.pow(subs, 2), axis=1)
    error_per_image_reg = tf.multiply(error_per_image, _regularize(labels))
    return tf.reduce_sum(error_per_image_reg)


def sm_cross_loss(logits, labels):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)


def _regularize(labels, pos_multiplier=1.5, neg_multiplier=0.5):
    tf_positive_label = tf.constant([0, 1], dtype=tf.float32)
    tf_mask = tf.equal(tf_positive_label, labels)
    tf_mask = tf.reduce_all(tf_mask, axis=1)

    batch_size = labels.shape[0]
    tf_pos_multiplier = tf.multiply(tf.ones([batch_size], dtype=tf.float32), pos_multiplier)
    tf_neg_multiplier = tf.multiply(tf.ones([batch_size], dtype=tf.float32), neg_multiplier)

    regularization_multipliers = tf.where(tf_mask, tf_pos_multiplier, tf_neg_multiplier)
    return regularization_multipliers
