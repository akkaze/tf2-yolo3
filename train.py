from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, CSVLogger
from yolo3.models import YoloV3, YoloV3Tiny, YoloLoss, yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors, yolo_tiny_anchor_masks
import yolo3.dataset as dataset
import sys

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_boolean('eager', True, 'run_eagerly or not')
flags.DEFINE_boolean('m2nist', True, 'use m2nist datast or not')
flags.DEFINE_string('checkpoint', '', 'checkpoint file for resume training')
flags.DEFINE_list('size', [64, 80], 'image size')
flags.DEFINE_integer('epochs', 30, 'number of epochs')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_float('learning_rate', 1e-4, 'learning rate')
flags.DEFINE_integer('num_classes', 10, 'number of classes in the model')


def main(_argv):
    size = (int(FLAGS.size[0]), int(FLAGS.size[1]))
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    os.makedirs('checkpoints', exist_ok=True)
    if FLAGS.tiny:
        if FLAGS.m2nist:
            channels = 1
        else:
            channels = 3
        model = YoloV3Tiny(size, channels=channels, training=True, num_classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(size, training=True, num_classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks
    if FLAGS.checkpoint:
        model.load_weights(FLAGS.checkpoint)
        logging.info('resume training')

    if not FLAGS.m2nist:
        train_dataset = dataset.load_textline_dataset(FLAGS.dataset, size)

        if FLAGS.val_dataset:
            val_dataset = dataset.load_textline_dataset(FLAGS.val_dataset, size)

    else:
        train_dataset, val_dataset = dataset.load_m2nist_dataset(size[::-1], 0.2)

    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(
        lambda x, y: (x, dataset.transform_targets(y, size, anchors, anchor_masks, FLAGS.num_classes, FLAGS.tiny)))
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(
        lambda x, y: (x, dataset.transform_targets(y, size, anchors, anchor_masks, FLAGS.num_classes, FLAGS.tiny)))
    model.summary()
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], num_classes=FLAGS.num_classes) for mask in anchor_masks]
    model.compile(optimizer=optimizer, loss=loss, run_eagerly=FLAGS.eager)

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_{epoch}.h5', verbose=1, save_weights_only=False),
        CSVLogger('training.log')
    ]

    history = model.fit(train_dataset, epochs=FLAGS.epochs, callbacks=callbacks, validation_data=None)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
