from functools import partial
import tensorflow as tf

_EPSILON = tf.keras.backend.epsilon()


def register_keras_custom_object(cls):
    tf.keras.utils.get_custom_objects()[cls.__name__] = cls
    return cls


def binary_focal_loss(y_true, y_pred, gamma, *, pos_weight=None, from_logits=False, label_smoothing=None):
    y_pred = tf.convert_to_tensor(y_pred)
    if not y_pred.dtype.is_floating:
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.float32)

    if from_logits:
        return _binary_focal_loss_from_logits(labels=y_true,
                                              logits=y_pred,
                                              gamma=gamma,
                                              pos_weight=pos_weight,
                                              label_smoothing=label_smoothing)
    else:
        return _binary_focal_loss_from_probs(labels=y_true,
                                             p=y_pred,
                                             gamma=gamma,
                                             pos_weight=pos_weight,
                                             label_smoothing=label_smoothing)


@register_keras_custom_object
class BinaryFocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma, *, pos_weight=None, from_logits=False, label_smoothing=None, **kwargs):

        super().__init__(**kwargs)
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.from_logits = from_logits
        self.label_smoothing = label_smoothing

    def get_config(self):
        config = super().get_config()
        config.update(gamma=self.gamma,
                      pos_weight=self.pos_weight,
                      from_logits=self.from_logits,
                      label_smoothing=self.label_smoothing)
        return config

    def call(self, y_true, y_pred):
        return binary_focal_loss(y_true=y_true,
                                 y_pred=y_pred,
                                 gamma=self.gamma,
                                 pos_weight=self.pos_weight,
                                 from_logits=self.from_logits,
                                 label_smoothing=self.label_smoothing)


# Helper functions below


def _process_labels(labels, label_smoothing, dtype):
    labels = tf.dtypes.cast(labels, dtype=dtype)
    if label_smoothing is not None:
        labels = (1 - label_smoothing) * labels + label_smoothing * 0.5
    return labels


def _binary_focal_loss_from_logits(labels, logits, gamma, pos_weight, label_smoothing):
    labels = _process_labels(labels=labels, label_smoothing=label_smoothing, dtype=logits.dtype)

    # Compute probabilities for the positive class
    p = tf.math.sigmoid(logits)

    if label_smoothing is None:
        labels_shape = labels.shape
        logits_shape = logits.shape
        if not labels_shape.is_fully_defined() or labels_shape != logits_shape:
            labels_shape = tf.shape(labels)
            logits_shape = tf.shape(logits)
            shape = tf.broadcast_dynamic_shape(labels_shape, logits_shape)
            labels = tf.broadcast_to(labels, shape)
            logits = tf.broadcast_to(logits, shape)
        if pos_weight is None:
            loss_func = tf.nn.sigmoid_cross_entropy_with_logits
        else:
            loss_func = partial(tf.nn.weighted_cross_entropy_with_logits, pos_weight=pos_weight)
        loss = loss_func(labels=labels, logits=logits)
        modulation_pos = (1 - p)**gamma
        modulation_neg = p**gamma
        mask = tf.dtypes.cast(labels, dtype=tf.bool)
        modulation = tf.where(mask, modulation_pos, modulation_neg)
        return modulation * loss

    # Terms for the positive and negative class components of the loss
    pos_term = labels * ((1 - p)**gamma)
    neg_term = (1 - labels) * (p**gamma)

    # Term involving the log and ReLU
    log_weight = pos_term
    if pos_weight is not None:
        log_weight *= pos_weight
    log_weight += neg_term
    log_term = tf.math.log1p(tf.math.exp(-tf.math.abs(logits)))
    log_term += tf.nn.relu(-logits)
    log_term *= log_weight

    # Combine all the terms into the loss
    loss = neg_term * logits + log_term
    return loss


def _binary_focal_loss_from_probs(labels, p, gamma, pos_weight, label_smoothing):
    q = 1 - p

    # For numerical stability (so we don't inadvertently take the log of 0)
    p = tf.math.maximum(p, _EPSILON)
    q = tf.math.maximum(q, _EPSILON)

    # Loss for the positive examples
    pos_loss = -(q**gamma) * tf.math.log(p)
    if pos_weight is not None:
        pos_loss *= pos_weight

    # Loss for the negative examples
    neg_loss = -(p**gamma) * tf.math.log(q)

    # Combine loss terms
    if label_smoothing is None:
        labels = tf.dtypes.cast(labels, dtype=tf.bool)
        loss = tf.where(labels, pos_loss, neg_loss)
    else:
        labels = _process_labels(labels=labels, label_smoothing=label_smoothing, dtype=p.dtype)
        loss = labels * pos_loss + (1 - labels) * neg_loss

    return loss