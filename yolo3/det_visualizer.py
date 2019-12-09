import os
import cv2
import uuid
import numpy as np
import shutil
from tensorflow.keras.callbacks import Callback
from .models import yolo_anchors, yolo_anchor_masks, yolo_tiny_anchors, yolo_tiny_anchor_masks
from .convert import make_eval_model_from_trained_model
from .utils import draw_outputs, cv2_letterbox_resize


class DetVisualizer(Callback):
    def __init__(self, dataset, result_dir='dets', num_batches=64, tiny=True):
        self.result_dir = result_dir
        self.dataset = dataset
        self.num_batches = num_batches
        self.tiny = tiny
        super(DetVisualizer, self).__init__()

    def on_train_begin(self, logs=None):
        if os.path.exists(self.result_dir):
            shutil.rmtree(self.result_dir, ignore_errors=True)
        else:
            os.makedirs(self.result_dir)

    def on_epoch_end(self, epoch, logs=None):
        if self.tiny:
            anchors = yolo_tiny_anchors
            masks = yolo_tiny_anchor_masks
        else:
            anchors = yolo_anchors
            masks = yolo_anchor_masks
        model = make_eval_model_from_trained_model(self.model, anchors, masks)

        epoch_dir = os.path.join(self.result_dir, str(epoch))

        os.makedirs(epoch_dir)
        for batch, (images, labels) in enumerate(self.dataset):
            images = images.numpy()
            boxes, scores, classes = model.predict(images)
            for i in range(boxes.shape[0]):
                img_for_this = (images[i, ...] * 255).astype(np.uint8)

                boxes_for_this, scores_for_this, classes_for_this = boxes[i, ...], scores[i, ...], classes[i, ...]

                img_for_this = draw_outputs(img_for_this, (boxes_for_this, scores_for_this, classes_for_this))
                cv2.imwrite(os.path.join(epoch_dir, '{0}.jpg'.format(uuid.uuid4())), img_for_this)
            if batch == self.num_batches:
                break
