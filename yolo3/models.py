from absl import flags
from absl.flags import FLAGS
import numpy as np
import tensorflow as tf
from tensorflow.keras import Model
import cv2
from tensorflow.keras.layers import Add, Concatenate, Conv2D, Input, Lambda, LeakyReLU, UpSampling2D, \
    ZeroPadding2D, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.losses import binary_crossentropy, sparse_categorical_crossentropy
from .utils import broadcast_iou

yolo_anchors = np.array([(1 / 9., 3 / 9.), (2 / 9., 2 / 9.), (3 / 9., 1 / 9.), (2 / 9., 6 / 9.), (4 / 9., 4 / 9.),
                         (6 / 9., 2 / 9.), (3 / 9., 9 / 9.), (6 / 9., 6 / 9.), (9 / 9., 3 / 9.)], np.float32)
yolo_anchor_masks = np.array([[6, 7, 8], [3, 4, 5], [0, 1, 2]])

yolo_tiny_anchors = np.array([(1 / 6., 3 / 6.), (2 / 6., 2 / 6.), (3 / 6., 1 / 6.), (2 / 6., 6 / 6.), (4 / 6., 4 / 6.),
                              (
                                  6 / 6.,
                                  2 / 6,
                              )], np.float32)
yolo_tiny_anchor_masks = np.array([[3, 4, 5], [0, 1, 2]])


def DarknetConv(x, filters, size, strides=1, padding='same', batch_norm=True):
    x = Conv2D(filters=filters,
               kernel_size=size,
               strides=strides,
               padding=padding,
               use_bias=not batch_norm,
               kernel_regularizer=l2(0.0005))(x)
    if batch_norm:
        x = BatchNormalization()(x)
        x = LeakyReLU(alpha=0.1)(x)
    return x


def DarknetResidual(x, filters):
    prev = x
    x = DarknetConv(x, filters // 2, 1)
    x = DarknetConv(x, filters, 3)
    x = Add()([prev, x])
    return x


def DarknetBlock(x, filters, blocks):
    x = DarknetConv(x, filters, 3, strides=2)
    for _ in range(blocks):
        x = DarknetResidual(x, filters)
    return x


def Darknet(name=None, num_channels=3):
    x = inputs = Input([None, None, num_channels])
    x = DarknetConv(x, 16, 3)
    x = DarknetBlock(x, 32, 1)
    x = DarknetBlock(x, 64, 2)  # skip connection
    x = x_36 = DarknetBlock(x, 128, 8)  # skip connection
    x = x_61 = DarknetBlock(x, 256, 8)
    x = DarknetBlock(x, 512, 4)
    return tf.keras.Model(inputs, (x_36, x_61, x), name=name)


def DarknetTiny(name=None, num_channels=3):
    x = inputs = Input([None, None, num_channels])
    x = DarknetConv(x, 8, 3, 2)
    x = DarknetConv(x, 16, 3, 2)
    x = x_8 = DarknetConv(x, 24, 3, 2)
    x = DarknetConv(x, 48, 3, 2)
    x = DarknetConv(x, 64, 3)
    return tf.keras.Model(inputs, (x_8, x), name=name)


def YoloConv(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])

        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        x = DarknetConv(x, filters * 2, 3)
        x = DarknetConv(x, filters, 1)
        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloConvTiny(filters, name=None):
    def yolo_conv(x_in):
        if isinstance(x_in, tuple):
            inputs = Input(x_in[0].shape[1:]), Input(x_in[1].shape[1:])
            x, x_skip = inputs

            # concat with skip connection
            x = DarknetConv(x, filters, 1)
            x = UpSampling2D(2)(x)
            x = Concatenate()([x, x_skip])
        else:
            x = inputs = Input(x_in.shape[1:])
            x = DarknetConv(x, filters, 1)

        return Model(inputs, x, name=name)(x_in)

    return yolo_conv


def YoloOutput(filters, anchors, classes, name=None):
    def yolo_output(x_in):
        x = inputs = Input(x_in.shape[1:])
        x = DarknetConv(x, filters, 3)
        x = DarknetConv(x, anchors * (classes + 5), 1, batch_norm=False)
        x = Lambda(lambda x: tf.reshape(x, (-1, tf.shape(x)[1], tf.shape(x)[2], anchors, classes + 5)))(x)
        return tf.keras.Model(inputs, x, name=name)(x_in)

    return yolo_output


def yolo_boxes(pred, anchors, num_classes, training=True):
    # pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...classes))
    grid_size = tf.shape(pred)[1:2]
    grid_y, grid_x = tf.shape(pred)[1], tf.shape(pred)[2]

    box_xy, box_wh, objectness, class_probs = tf.split(pred, (2, 2, 1, num_classes), axis=-1)
    box_xy = tf.sigmoid(box_xy)

    objectness = tf.sigmoid(objectness)
    if training:
        class_probs = tf.nn.softmax(class_probs)
    else:
        class_probs = tf.cast(tf.equal(tf.reduce_max(class_probs, axis=-1, keepdims=True), class_probs), tf.float32)
    pred_box = tf.concat((box_xy, box_wh), axis=-1)  # original xywh for loss

    # !!! grid[x][y] == (y, x)
    grid = tf.meshgrid(tf.range(grid_x), tf.range(grid_y))
    grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)  # [gx, gy, 1, 2]

    box_xy = (box_xy + tf.cast(grid, tf.float32)) / tf.cast(grid_size, tf.float32)
    box_wh = tf.exp(box_wh) * anchors

    box_x1y1 = box_xy - box_wh / 2
    box_x2y2 = box_xy + box_wh / 2
    bbox = tf.concat([box_x1y1, box_x2y2], axis=-1)

    return bbox, objectness, class_probs, pred_box


def nms(bboxes, scores, iou_threshold):
    # If no bounding boxes, return empty list
    if len(bboxes) == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0), np.int32)
    boxes = np.array(bboxes)
    start_x, start_y, end_x, end_y = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]

    score = np.array(scores)
    # Picked bounding boxes
    picked_boxes, picked_scores = [], []

    # Compute areas of bounding boxes
    areas = (end_x - start_x + 1) * (end_y - start_y + 1)
    # Sort by confidence score of bounding boxes
    order = np.argsort(score)

    # Iterate bounding boxes
    while order.size > 0:
        # The index of largest confidence score
        index = order[-1]
        # Pick the bounding box with largest confidence score
        picked_boxes.append(bboxes[index])
        picked_scores.append(scores[index])
        # Compute ordinates of intersection-over-union(IOU)
        x1, x2 = np.maximum(start_x[index], start_x[order[:-1]]), np.minimum(end_x[index], end_x[order[:-1]])
        y1, y2 = np.maximum(start_y[index], start_y[order[:-1]]), np.minimum(end_y[index], end_y[order[:-1]])
        # Compute areas of intersection-over-union
        w, h = np.maximum(0.0, x2 - x1 + 1), np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        # Compute the iou
        iou = intersection / (areas[index] + areas[order[:-1]] - intersection)

        left = np.where(iou < iou_threshold)
        order = order[left]
    return np.stack(picked_boxes), np.stack(picked_scores)


def batched_nms(bboxes, scores, iou_threshold):
    bboxes, scores, iou_threshold = bboxes.numpy(), scores.numpy(), iou_threshold.numpy()
    picked_boxes, picked_scores = [], []
    for i in range(bboxes.shape[0]):
        bboxes_this_bacth = bboxes[i, ...]
        scores_this_batch = scores[i, ...]
        picked_boxes_this_batch, picked_scores_this_batch = nms(bboxes_this_bacth, scores_this_batch, iou_threshold)
        picked_boxes.append(picked_boxes_this_batch)
        picked_scores.append(picked_scores_this_batch)
    picked_boxes = np.stack(picked_boxes)
    picked_scores = np.stack(picked_scores)
    return picked_boxes, picked_scores


def yolo_nms(outputs, anchors, masks, num_classes, iou_threshold=0.5, score_threshold=0.5):
    boxes, confs, classes = [], [], []

    for o in outputs:
        boxes.append(tf.reshape(o[0], (tf.shape(o[0])[0], -1, tf.shape(o[0])[-1])))
        confs.append(tf.reshape(o[1], (tf.shape(o[0])[0], -1, tf.shape(o[1])[-1])))
        classes.append(tf.reshape(o[2], (tf.shape(o[0])[0], -1, tf.shape(o[2])[-1])))
    boxes = tf.concat(boxes, axis=1)
    confs = tf.concat(confs, axis=1)
    class_probs = tf.concat(classes, axis=1)
    box_scores = confs * class_probs
    mask = box_scores >= score_threshold

    boxes_ = []
    scores_ = []
    classes_ = []
    for c in range(num_classes):
        class_boxes = tf.boolean_mask(boxes, mask[..., c])
        class_boxes = tf.reshape(class_boxes, (1, -1, 4))

        class_box_scores = tf.boolean_mask(box_scores[..., c], mask[..., c])
        class_box_scores = tf.reshape(class_box_scores, (1, -1))

        class_boxes, class_box_scores = tf.py_function(func=batched_nms,
                                                       inp=[class_boxes, class_box_scores, iou_threshold],
                                                       Tout=[tf.float32, tf.float32])
        classes = tf.ones_like(class_box_scores, tf.int32) * c
        boxes_.append(class_boxes)
        scores_.append(class_box_scores)
        classes_.append(classes)
    boxes_ = tf.concat(boxes_, axis=1)
    scores_ = tf.concat(scores_, axis=1)
    classes_ = tf.concat(classes_, axis=1)

    return boxes_, scores_, classes_


def YoloV3(size=None, num_channels=3, anchors=yolo_anchors, masks=yolo_anchor_masks, num_classes=10, training=False):
    x = inputs = Input([*size, num_channels])

    x_36, x_61, x = Darknet(name='yolo_darknet', num_channels=num_channels)(x)

    x = YoloConv(512, name='yolo_conv_0')(x)
    output_0 = YoloOutput(512, len(masks[0]), num_classes, name='yolo_output_0')(x)

    x = YoloConv(256, name='yolo_conv_1')((x, x_61))
    output_1 = YoloOutput(256, len(masks[1]), num_classes, name='yolo_output_1')(x)

    x = YoloConv(128, name='yolo_conv_2')((x, x_36))
    output_2 = YoloOutput(128, len(masks[2]), num_classes, name='yolo_output_2')(x)

    if training:
        return Model(inputs, (output_0, output_1, output_2), name='yolov3')

    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes), name='yolo_boxes_0')(output_0)
    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes), name='yolo_boxes_1')(output_1)
    boxes_2 = Lambda(lambda x: yolo_boxes(x, anchors[masks[2]], num_classes), name='yolo_boxes_2')(output_2)

    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, num_classes),
                     name='yolo_nms')((boxes_0[:3], boxes_1[:3], boxes_2[:3]))

    return Model(inputs, outputs, name='yolov3')


def YoloV3Tiny(size=None,
               num_channels=3,
               anchors=yolo_tiny_anchors,
               masks=yolo_tiny_anchor_masks,
               num_classes=10,
               training=False):
    x = inputs = Input([*size, num_channels])

    x_8, x = DarknetTiny(name='yolo_darknet', num_channels=num_channels)(x)

    x = YoloConvTiny(128, name='yolo_conv_0')(x)
    output_0 = YoloOutput(128, len(masks[0]), num_classes, name='yolo_output_0')(x)

    x = YoloConvTiny(64, name='yolo_conv_1')((x, x_8))
    output_1 = YoloOutput(64, len(masks[1]), num_classes, name='yolo_output_1')(x)
    if training:
        return Model(inputs, (output_0, output_1), name='yolov3')
    boxes_0 = Lambda(lambda x: yolo_boxes(x, anchors[masks[0]], num_classes), name='yolo_boxes_0')(output_0)

    boxes_1 = Lambda(lambda x: yolo_boxes(x, anchors[masks[1]], num_classes), name='yolo_boxes_1')(output_1)
    outputs = Lambda(lambda x: yolo_nms(x, anchors, masks, num_classes), name='yolo_nms')((boxes_0[:3], boxes_1[:3]))
    return Model(inputs, outputs, name='yolov3_tiny')


def YoloLoss(anchors, num_classes=10, ignore_thresh=0.5):
    def yolo_loss(y_true, y_pred):
        # 1. transform all pred outputs
        # y_pred: (batch_size, grid, grid, anchors, (x, y, w, h, obj, ...cls))
        pred_box, pred_obj, pred_class, pred_xywh = yolo_boxes(y_pred, anchors, num_classes)
        pred_xy = pred_xywh[..., 0:2]
        pred_wh = pred_xywh[..., 2:4]

        # 2. transform all true outputs
        # y_true: (batch_size, grid, grid, anchors, (x1, y1, x2, y2, obj, cls))
        true_box, true_obj, true_class_idx = tf.split(y_true, (4, 1, 1), axis=-1)
        true_xy = (true_box[..., 0:2] + true_box[..., 2:4]) / 2.
        true_wh = true_box[..., 2:4] - true_box[..., 0:2]

        # true_obj = tf.cast(true_obj, tf.int32)
        # true_class_idx = tf.cast(true_class_idx, tf.int32)
        # tf.print(tf.math.count_nonzero(true_class_idx))
        # give higher weights to small boxes
        box_loss_scale = 2 - true_wh[..., 0] * true_wh[..., 1]

        # 3. inverting the pred box equations
        grid_size = tf.shape(y_pred)[1:2]

        grid_y, grid_x = tf.shape(y_pred)[1], tf.shape(y_pred)[2]
        grid = tf.meshgrid(tf.range(grid_x), tf.range(grid_y))

        grid = tf.expand_dims(tf.stack(grid, axis=-1), axis=2)
        #tf.print(tf.math.count_nonzero(true_xy))
        true_xy = true_xy * tf.cast(grid_size, tf.float32) - tf.cast(grid, tf.float32)
        true_wh = tf.math.log(true_wh / anchors)
        true_wh = tf.where(tf.math.is_inf(true_wh), tf.zeros_like(true_wh), true_wh)
        # 4. calculate all masks
        obj_mask = tf.squeeze(true_obj, -1)
        # ignore false positive when iou is over threshold
        true_box_flat = tf.boolean_mask(true_box, tf.cast(obj_mask, tf.bool))
        best_iou = tf.reduce_max(broadcast_iou(pred_box, true_box_flat), axis=-1)
        ignore_mask = tf.cast(best_iou < ignore_thresh, tf.float32)
        # 5. calculate all losses
        xy_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_xy - pred_xy), axis=-1)
        wh_loss = obj_mask * box_loss_scale * tf.reduce_sum(tf.square(true_wh - pred_wh), axis=-1)
        obj_loss = binary_crossentropy(true_obj, pred_obj)
        obj_loss = obj_mask * obj_loss + (1 - obj_mask) * ignore_mask * obj_loss
        class_loss = obj_mask * sparse_categorical_crossentropy(true_class_idx, pred_class)
        # 6. sum over (batch, gridx, gridy, anchors) => (batch, 1)
        xy_loss = tf.reduce_sum(xy_loss, axis=(1, 2, 3))
        wh_loss = tf.reduce_sum(wh_loss, axis=(1, 2, 3))
        obj_loss = tf.reduce_sum(obj_loss, axis=(1, 2, 3))
        class_loss = tf.reduce_sum(class_loss, axis=(1, 2, 3))
        loss = xy_loss + wh_loss + obj_loss + class_loss
        return loss

    return yolo_loss