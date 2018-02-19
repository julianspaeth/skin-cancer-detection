import tensorflow as tf

def l1_loss(logits=None, labels=None):
    subs = tf.subtract(labels, logits)
    return tf.reduce_sum(tf.abs(subs))

def l2_loss(logits=None, labels=None):
    subs = tf.subtract(labels, logits)
    return tf.reduce_sum(tf.pow(subs, 2))

def sm_cross_loss(logits=None, labels=None):
    return tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)

# def _regularize(labels=None):
#     batch_size = labels.shape[0]
#     ret = np.ones([batch_size])