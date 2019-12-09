import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Lambda
from yolo3.models import YoloV3, YoloV3Tiny, YoloLoss, yolo_nms, yolo_boxes, yolo_anchors, \
    yolo_anchor_masks, yolo_tiny_anchors, yolo_tiny_anchor_masks
from yolo3.convert import make_eval_model_from_trained_model
from yolo3.utils import draw_outputs, cv2_letterbox_resize
from yolo3.convert import make_eval_model_from_trained_model

flags.DEFINE_string('weights', './checkpoints/yolov3_22.h5', 'path to weights file')
flags.DEFINE_string('checkpoint', '', 'path to checkpoint file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_list('size', [64, 96], 'resize images to')
flags.DEFINE_string('image', './data/4000.png', 'path to input image')
flags.DEFINE_string('output', './output.png', 'path to output image')
flags.DEFINE_integer('num_classes', 10, 'number of classes in the model')
flags.DEFINE_integer('num_channels', 1, 'number of channels of image')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    size = (int(FLAGS.size[0]), int(FLAGS.size[1]))

    yolo = tf.keras.models.load_model(FLAGS.weights,
                                      custom_objects={'yolo_loss': YoloLoss(np.zeros((3, 2), np.float32))})
    if FLAGS.checkpoint:
        if FLAGS.tiny:
            anchors = yolo_tiny_anchors
            masks = yolo_tiny_anchor_masks
        else:
            anchors = yolo_anchors
            masks = yolo_anchor_masks
        yolo = make_eval_model_from_trained_model(yolo, anchors, masks, FLAGS.tiny)
        logging.info('checkpoint loaded')
    else:
        if FLAGS.tiny:
            yolo = YoloV3Tiny(size, num_channels=FLAGS.num_channels, num_classes=FLAGS.num_classes)
        else:
            yolo = YoloV3(size, num_channels=FLAGS.num_channels, classes=FLAGS.num_classes)
        yolo.load_weights(FLAGS.weights)
        logging.info('weights loaded')
    yolo.summary()

    img = cv2.imread(FLAGS.image, cv2.IMREAD_UNCHANGED)
    img, _ = cv2_letterbox_resize(img, size[::-1])
    img_ = img / 255.
    img_ = np.expand_dims(img_, 0)

    t1 = time.time()
    boxes, scores, classes = yolo.predict(img_)
    print(scores)
    t2 = time.time()
    logging.info('time: {}'.format(t2 - t1))

    logging.info('detections:')
    for i in range(boxes.shape[1]):
        logging.info('\t{}, {}, {}'.format(int(classes[0][i]), np.array(scores[0][i]), np.array(boxes[0][i])))

    img = draw_outputs(img, (boxes[0], scores[0], classes[0]))
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
