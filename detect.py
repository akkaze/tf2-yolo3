import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2
import numpy as np
import tensorflow as tf
from yolo3.models import YoloV3, YoloV3Tiny
from yolo3.dataset import transform_images
from yolo3.utils import draw_outputs, cv2_letterbox_resize

flags.DEFINE_string('weights', './checkpoints/yolov3_8.h5', 'path to weights file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_list('size', [64, 64], 'resize images to')
flags.DEFINE_string('image', './data/0.png', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 10, 'number of classes in the model')


def main(_argv):
    physical_devices = tf.config.experimental.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    size = (int(FLAGS.size[0]), int(FLAGS.size[1]))
    if FLAGS.tiny:
        yolo = YoloV3Tiny(size, channels=1, num_classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(classes=FLAGS.num_classes)
    yolo.load_weights(FLAGS.weights)
    logging.info('weights loaded')

    img = cv2.imread(FLAGS.image, cv2.IMREAD_UNCHANGED)
    img, _ = cv2_letterbox_resize(img, size)
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

    # img = cv2.resize(img, (FLAGS.size, FLAGS.size))
    img = draw_outputs(img, (boxes, scores, classes))
    cv2.imwrite(FLAGS.output, img)
    logging.info('output saved to: {}'.format(FLAGS.output))


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
