from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.callbacks import (ReduceLROnPlateau, EarlyStopping, ModelCheckpoint, TensorBoard)
from yolo3.models import (YoloV3, YoloV3Tiny, YoloLoss, yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors,
                          yolo_tiny_anchor_masks)

import yolo3.dataset as dataset
import sys
flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_string('val_dataset', '', 'path to validation dataset')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_enum('eager', True, 'run_eagerly or not')
flags.DEFINE_string('checkpoint', '', 'checkpoint file for resume training')
flags.DEFINE_integer('size', 256, 'image size')
flags.DEFINE_integer('epochs', 29, 'number of epochs')
flags.DEFINE_integer('batch_size', 1, 'batch size')
flags.DEFINE_float('learning_rate', 1e-1, 'learning rate')
flags.DEFINE_integer('num_classes', 20, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    train_dataset = dataset.load_textline_dataset(FLAGS.dataset, FLAGS.size)
    train_dataset = train_dataset.shuffle(buffer_size=1024)
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y:
                                      (x, dataset.transform_targets(y, anchors, anchor_masks, FLAGS.num_classes)))

    train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    if FLAGS.val_dataset:
        val_dataset = dataset.load_textline_dataset(FLAGS.val_dataset, FLAGS.size)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y:
                                  (x, dataset.transform_targets(y, anchors, anchor_masks, FLAGS.num_classes)))

    model.summary()
    #model.save('backbone.h5')
    if FLAGS.checkpoint:
        model.load(FLAGS.checkpoint)
    #sys.exit()
    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes) for mask in anchor_masks]

    model.compile(optimizer=optimizer, loss=loss, run_eagerly=(FLAGS.mode == 'eager_fit'))

    callbacks = [
        ReduceLROnPlateau(verbose=1),
        EarlyStopping(patience=3, verbose=1),
        ModelCheckpoint('checkpoints/yolov3_train_{epoch}.h5', verbose=1, save_weights_only=True),
        TensorBoard(log_dir='logs')
    ]

    history = model.fit(train_dataset, epochs=FLAGS.epochs, callbacks=callbacks, validation_data=val_dataset)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
